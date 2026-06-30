from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.models.callbacks import (
    BucketStats,
    ClassConditionalSampleCallback,
    ColouriserLPIPSCallback,
    ColouriserSamplingSweepCallback,
    EMAWeightAveraging,
    FIDCallback,
    InputGridCallback,
    MLFlowArtifactCallback,
    RunningABStats,
    UnconditionalDebugCallback,
    UnconditionalSampleCallback,
    _flow_matching_step_uncond,
    _generate_samples,
    _get_mlflow_logger,
    _get_model_checkpoint_callback,
    _get_sampling_metadata,
    _log_bucket_histograms,
    _log_colouriser_ab_histogram,
    _log_sample_image,
    _mlflow_run_context,
)
from flow_matching.models.config import UNetConfig, make_optimizer
from flow_matching.models.unet import (
    ClassCondUNet,
    LEncoderColouriser,
    SimpleColouriser,
    UNet,
)
from flow_matching.utils.train import flow_matching_step, flow_matching_step_cfg, unpack_batch

class BaseFlowMatchingModule(pl.LightningModule):
    is_conditional = False
    model_artifact_name = "UNet"

    def __init__(
        self,
        unet_cfg: dict[str, Any] | None = None,
        optimizer_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.unet_cfg = UNetConfig(**(unet_cfg or {}))
        self.optimizer_name = str(optimizer_name)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.loss_fn = nn.MSELoss()
        self.generator = self._build_generator()

    def _build_generator(self) -> nn.Module:
        raise NotImplementedError

    def configure_optimizers(self):
        return make_optimizer(
            self.parameters(),
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def _batch_size(self, batch) -> int:
        images, _ = unpack_batch(batch)
        return int(images.shape[0])

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        batch_size = self._batch_size(batch)
        self.log(
            "train_mse_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_mse_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        batch_size = self._batch_size(batch)
        self.log(
            "val_mse_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss


class UnconditionalFlowMatchingModule(BaseFlowMatchingModule):
    is_conditional = False
    model_artifact_name = "UNet"

    def _build_generator(self) -> nn.Module:
        return UNet.from_config(self.unet_cfg)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.generator(x, t)

    def _loss(self, batch) -> torch.Tensor:
        x, _ = unpack_batch(batch)
        return flow_matching_step(self.generator, x, self.loss_fn, self.device)


class ClassConditionalFlowMatchingModule(BaseFlowMatchingModule):
    is_conditional = True
    model_artifact_name = "ClassCondUNet"

    def __init__(
        self,
        unet_cfg: dict[str, Any] | None = None,
        optimizer_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        n_classes: int = 10,
        d_cls_emb: int = 128,
        null_id: int | None = None,
        p_drop: float = 0.2,
    ) -> None:
        self.n_classes = int(n_classes)
        self.null_id = int(n_classes if null_id is None else null_id)
        if self.null_id < 0:
            raise ValueError(f"null_id must be >= 0, got {self.null_id}")
        self.class_vocab_size = max(self.n_classes, self.null_id + 1)
        self.d_cls_emb = int(d_cls_emb)
        self.p_drop = float(p_drop)
        super().__init__(
            unet_cfg=unet_cfg,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters()

    def _build_generator(self) -> nn.Module:
        core = UNet.from_config(self.unet_cfg)
        return ClassCondUNet(
            core=core, n_classes=self.class_vocab_size, d_cls_emb=self.d_cls_emb
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.generator(x, t, y)

    def _loss(self, batch) -> torch.Tensor:
        x, y = unpack_batch(batch)
        if y is None:
            raise ValueError(
                "ClassConditionalFlowMatchingModule requires labels. Set datamodule.drop_labels=False."
            )
        return flow_matching_step_cfg(
            self.generator,
            x,
            y,
            self.p_drop,
            self.null_id,
            self.loss_fn,
            self.device,
        )


class SimpleColouriserFlowMatchingModule(BaseFlowMatchingModule):
    model_artifact_name = "SimpleColouriser"

    def __init__(
        self,
        unet_cfg: dict[str, Any] | None = None,
        optimizer_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__(
            unet_cfg=unet_cfg,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop must be in [0, 1), got {p_drop}")
        self.p_drop = float(p_drop)
        self.save_hyperparameters()

    def _build_generator(self) -> nn.Module:
        return SimpleColouriser.from_config(self.unet_cfg)

    def forward(
        self, ab_t: torch.Tensor, t: torch.Tensor, L: torch.Tensor
    ) -> torch.Tensor:
        return self.generator(ab_t, t, L)

    def _loss(self, batch) -> torch.Tensor:
        LAB, _ = unpack_batch(batch)
        LAB = LAB.to(self.device)
        L = LAB[:, :1, ...]
        ab = LAB[:, 1:, ...]
        B = ab.shape[0]
        x0 = torch.randn_like(ab)
        t = torch.rand(B, 1, 1, 1, device=ab.device, dtype=ab.dtype)
        ab_t = (1 - t) * x0 + t * ab
        if self.p_drop > 0:
            drop_mask = (
                torch.rand((B, 1, 1, 1), device=ab.device, dtype=ab.dtype) < self.p_drop
            )
            L = torch.where(drop_mask, torch.zeros_like(L), L)
        v_est = self.generator(ab_t, t, L)
        v_true = ab - x0
        return self.loss_fn(v_est, v_true)


class LEncoderColouriserFlowMatchingModule(SimpleColouriserFlowMatchingModule):
    model_artifact_name = "LEncoderColouriser"

    def _build_generator(self) -> nn.Module:
        return LEncoderColouriser.from_config(self.unet_cfg)
