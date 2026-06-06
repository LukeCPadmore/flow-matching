from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from models.config import OptimConfig, UNetConfig
from models.ode_solvers import (
    get_ode_solver_from_name,
    sample_conditional,
    sample_unconditional,
)
from models.unet import ClassCondUNet, UNet
from utils.FID.fid_evaluation import build_fid_metric
from utils.FID.fid_lightning import FIDClassifierLightningModule
from utils.train import create_pil_image, flow_matching_step, flow_matching_step_cfg


def _generate_samples(
    pl_module: pl.LightningModule,
    *,
    n_images: int,
    image_shape,
    n_steps: int,
    guidance_scale: float,
    ode_solver,
    sample_seed: int | None,
):
    if getattr(pl_module, "is_conditional", False):
        num_classes = int(getattr(pl_module, "n_classes"))
        labels = torch.arange(num_classes, device=pl_module.device)
        repeats = max(1, -(-n_images // num_classes))
        labels = labels.repeat(repeats)[:n_images]
        samples = sample_conditional(
            model=pl_module.generator,
            y=labels,
            image_shape=image_shape,
            ode_solver=ode_solver,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            null_id=getattr(pl_module, "null_id", None),
            return_all=False,
            device=pl_module.device,
            seed=sample_seed,
        )
        return samples, labels

    samples = sample_unconditional(
        model=pl_module.generator,
        n_images=n_images,
        image_shape=image_shape,
        ode_solver=ode_solver,
        n_steps=n_steps,
        return_all=False,
        device=pl_module.device,
        seed=sample_seed,
    )
    return samples, None


class BaseFlowMatchingModule(pl.LightningModule):
    is_conditional = False
    model_artifact_name = "UNet"

    def __init__(
        self,
        unet_cfg: dict[str, Any] | None = None,
        optim_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.unet_cfg = UNetConfig(**(unet_cfg or {}))
        self.optim_cfg = OptimConfig(**(optim_cfg or {}))
        self.loss_fn = nn.MSELoss()
        self.generator = self._build_generator()

    def _build_generator(self) -> nn.Module:
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optim_cfg.make_optimizer(self.parameters())

    def _loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def _batch_size(self, batch) -> int:
        return int(batch[0].shape[0])

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
        x, _ = batch
        return flow_matching_step(self.generator, x, self.loss_fn, self.device)


class ClassConditionalFlowMatchingModule(BaseFlowMatchingModule):
    is_conditional = True
    model_artifact_name = "ClassCondUNet"

    def __init__(
        self,
        unet_cfg: dict[str, Any] | None = None,
        optim_cfg: dict[str, Any] | None = None,
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
        super().__init__(unet_cfg=unet_cfg, optim_cfg=optim_cfg)
        self.save_hyperparameters()

    def _build_generator(self) -> nn.Module:
        core = UNet.from_config(self.unet_cfg)
        return ClassCondUNet(core=core, n_classes=self.class_vocab_size, d_cls_emb=self.d_cls_emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.generator(x, t, y)

    def _loss(self, batch) -> torch.Tensor:
        x, y = batch
        return flow_matching_step_cfg(
            self.generator,
            x,
            y,
            self.p_drop,
            self.null_id,
            self.loss_fn,
            self.device,
        )


class FlowMatchingSampleCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 5,
        n_images: int = 64,
        nrow: int = 8,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        ode_solver_name: str = "euler_solver",
        sample_seed: int | None = 0,
        artifact_prefix: str = "samples",
    ) -> None:
        self.every_n_epochs = int(every_n_epochs)
        self.n_images = int(n_images)
        self.nrow = int(nrow)
        self.n_steps = int(n_steps)
        self.guidance_scale = float(guidance_scale)
        self.ode_solver_name = ode_solver_name
        self.sample_seed = sample_seed
        self.artifact_prefix = artifact_prefix

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("FlowMatchingSampleCallback requires a datamodule with image_shape.")

        image_shape = getattr(datamodule, "image_shape", None)
        if image_shape is None:
            raise RuntimeError("Datamodule must expose image_shape for sampling callbacks.")

        sample_mean = getattr(datamodule, "sample_mean", None)
        sample_std = getattr(datamodule, "sample_std", None)
        ode_solver = get_ode_solver_from_name(self.ode_solver_name)

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            samples, labels = _generate_samples(
                pl_module,
                n_images=self.n_images,
                image_shape=image_shape,
                n_steps=self.n_steps,
                guidance_scale=self.guidance_scale,
                ode_solver=ode_solver,
                sample_seed=self.sample_seed,
            )
            image = create_pil_image(
                samples,
                nrow=self.nrow,
                labels=labels,
                mean=sample_mean,
                std=sample_std,
            )

        if was_training:
            pl_module.train()

        mlflow.log_image(
            image,
            artifact_file=f"{self.artifact_prefix}/epoch_{trainer.current_epoch:04d}.png",
        )


class InputGridCallback(Callback):
    def __init__(
        self,
        n_images: int = 64,
        nrow: int = 8,
        artifact_prefix: str = "inputs",
    ) -> None:
        self.n_images = int(n_images)
        self.nrow = int(nrow)
        self.artifact_prefix = artifact_prefix

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("InputGridCallback requires a datamodule.")

        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        if isinstance(batch, (tuple, list)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None

        sample_mean = getattr(datamodule, "sample_mean", None)
        sample_std = getattr(datamodule, "sample_std", None)
        image = create_pil_image(
            images[: self.n_images],
            nrow=self.nrow,
            labels=labels[: self.n_images] if labels is not None else None,
            mean=sample_mean,
            std=sample_std,
        )
        mlflow.log_image(image, artifact_file=f"{self.artifact_prefix}/train_start.png")


class FIDCallback(Callback):
    def __init__(
        self,
        backbone_run_id: str,
        *,
        backbone_artifact_path: str = "checkpoints/best.ckpt",
        every_n_epochs: int = 5,
        n_images: int = 64,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        ode_solver_name: str = "euler_solver",
        sample_seed: int | None = 0,
    ) -> None:
        self.backbone_run_id = backbone_run_id
        self.backbone_artifact_path = backbone_artifact_path
        self.every_n_epochs = int(every_n_epochs)
        self.n_images = int(n_images)
        self.n_steps = int(n_steps)
        self.guidance_scale = float(guidance_scale)
        self.ode_solver_name = ode_solver_name
        self.sample_seed = sample_seed
        self._fid_metric = None
        self._backbone = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if trainer.datamodule is None:
            raise RuntimeError("FIDCallback requires a datamodule.")

        self._backbone = FIDClassifierLightningModule.load_backbone_from_mlflow_run(
            self.backbone_run_id,
            artifact_path=self.backbone_artifact_path,
            map_location="cpu",
        )
        self._fid_metric = build_fid_metric(self._backbone)

        loader = trainer.datamodule.val_dataloader()
        with torch.no_grad():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                self._fid_metric.update(images, real=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self._fid_metric is None or self._backbone is None:
            raise RuntimeError("FIDCallback was not initialised correctly.")
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("FIDCallback requires a datamodule.")

        image_shape = getattr(datamodule, "image_shape", None)
        if image_shape is None:
            raise RuntimeError("Datamodule must expose image_shape for FID sampling.")

        ode_solver = get_ode_solver_from_name(self.ode_solver_name)
        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            samples, _ = _generate_samples(
                pl_module,
                n_images=self.n_images,
                image_shape=image_shape,
                n_steps=self.n_steps,
                guidance_scale=self.guidance_scale,
                ode_solver=ode_solver,
                sample_seed=self.sample_seed,
            )
            self._fid_metric.update(samples.cpu(), real=False)
            fid = float(self._fid_metric.compute().detach().cpu().item())
            self._fid_metric.reset()

        if was_training:
            pl_module.train()

        mlflow.log_metric("fid", fid, step=trainer.current_epoch)


class MLFlowArtifactCallback(Callback):
    def __init__(self, model_artifact_name: str) -> None:
        self.model_artifact_name = model_artifact_name

    def _find_config_file(self, trainer: pl.Trainer) -> Path | None:
        candidates: list[Path] = []
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir is not None:
            candidates.append(Path(log_dir) / "config.yaml")
        candidates.append(Path(trainer.default_root_dir) / "config.yaml")
        for path in candidates:
            if path.exists():
                return path
        return None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        if best_model_path:
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        last_model_path = getattr(checkpoint_callback, "last_model_path", "")
        if last_model_path:
            mlflow.log_artifact(last_model_path, artifact_path="checkpoints")

        config_path = self._find_config_file(trainer)
        if config_path is not None:
            mlflow.log_artifact(str(config_path), artifact_path="configs")

        best_score = getattr(checkpoint_callback, "best_model_score", None)
        if best_score is not None:
            mlflow.log_metric("best_mse", float(best_score.detach().cpu().item()))

        mlflow.pytorch.log_model(
            getattr(pl_module, "generator"),
            artifact_path=self.model_artifact_name,
        )
