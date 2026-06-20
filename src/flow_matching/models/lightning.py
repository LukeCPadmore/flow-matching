from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from src.flow_matching.models.config import UNetConfig, make_optimizer
from src.flow_matching.models.ode_solvers import (
    get_ode_solver_from_name,
    sample_conditional,
    sample_unconditional,
)
from src.flow_matching.models.unet import ClassCondUNet, UNet, SimpleColouriser
from src.flow_matching.utils.FID.fid_evaluation import build_fid_metric
from src.flow_matching.utils.FID.fid_lightning import FIDClassifierLightningModule
from src.flow_matching.utils.train import (
    create_pil_image,
    flow_matching_step,
    flow_matching_step_cfg,
    unpack_batch,
)


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


@dataclass
class BucketStats:
    mse_sums: torch.Tensor
    cossim_sums: torch.Tensor
    log_norm_ratio_sums: torch.Tensor
    freqs: torch.Tensor

    @classmethod
    def zeros(cls, num_buckets: int, device: torch.device) -> "BucketStats":
        z = torch.zeros(num_buckets, device=device, dtype=torch.float32)
        return cls(z.clone(), z.clone(), z.clone(), z.clone())

    def update(
        self,
        mse_batch: torch.Tensor,
        cossim_batch: torch.Tensor,
        log_norm_ratio_batch: torch.Tensor,
        bins: torch.Tensor,
    ) -> None:
        self.mse_sums += torch.bincount(
            bins, weights=mse_batch, minlength=self.freqs.numel()
        )
        self.cossim_sums += torch.bincount(
            bins, weights=cossim_batch, minlength=self.freqs.numel()
        )
        self.log_norm_ratio_sums += torch.bincount(
            bins, weights=log_norm_ratio_batch, minlength=self.freqs.numel()
        )
        self.freqs += torch.bincount(bins, minlength=self.freqs.numel()).to(
            self.freqs.dtype
        )

    def means(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe = self.freqs.clamp_min(1.0)
        mse = torch.where(
            self.freqs > 0, self.mse_sums / safe, torch.zeros_like(self.mse_sums)
        )
        cossim = torch.where(
            self.freqs > 0, self.cossim_sums / safe, torch.zeros_like(self.cossim_sums)
        )
        log_ratio = torch.where(
            self.freqs > 0,
            self.log_norm_ratio_sums / safe,
            torch.zeros_like(self.log_norm_ratio_sums),
        )
        return mse, cossim, log_ratio

    def add_(self, other: "BucketStats") -> None:
        self.mse_sums += other.mse_sums
        self.cossim_sums += other.cossim_sums
        self.log_norm_ratio_sums += other.log_norm_ratio_sums
        self.freqs += other.freqs


def _flow_matching_step_uncond(model, x1, loss_fn, device):
    x1 = x1.to(device)
    batch_size = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, 1, 1, 1, device=device)
    xt = (1 - t) * x0 + t * x1

    v_est = model(xt, t)
    v_true = x1 - x0
    mse_batch = loss_fn(v_est, v_true).mean(dim=(1, 2, 3))
    return mse_batch, t, v_est, v_true


def _log_bucket_histograms(
    stats: BucketStats, *, epoch: int, step: int, artifact_prefix: str
) -> None:
    mse_m, cos_m, log_m = stats.means()
    bucket_idx = torch.arange(stats.freqs.numel())
    plots = {
        "mse": mse_m,
        "cosine_similarity": cos_m,
        "log_norm_ratio": log_m,
        "bin_frequency": stats.freqs,
    }

    for name, values in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bucket_idx.cpu().numpy(), values.detach().cpu().numpy())
        ax.set_title(f"epoch {epoch:03d} {name}")
        ax.set_xlabel("t bucket")
        ax.set_ylabel(name)
        ax.set_xticks(bucket_idx.cpu().numpy())
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
        mlflow.log_image(
            img,
            artifact_file=f"{artifact_prefix}/{name}_epoch_{epoch:04d}_step_{step:06d}.png",
        )
        plt.close(fig)


class UnconditionalDebugCallback(Callback):
    def __init__(
        self,
        *,
        every_n_steps: int = 100,
        num_buckets: int = 20,
        artifact_prefix: str = "debug/unconditional",
    ) -> None:
        self.every_n_steps = int(every_n_steps)
        self.num_buckets = int(num_buckets)
        self.artifact_prefix = artifact_prefix
        self._loss_fn = nn.MSELoss(reduction="none")
        self._boundaries = torch.linspace(0, 1, self.num_buckets - 1)
        self._epoch_stats: BucketStats | None = None
        self._overall_stats: BucketStats | None = None

    def _maybe_log_summary(
        self, *, stats: BucketStats, epoch: int, step: int, suffix: str
    ) -> None:
        mse_m, cos_m, log_m = stats.means()
        for i, v in enumerate(mse_m.detach().cpu().tolist()):
            mlflow.log_metric(f"{suffix}_mse_bucket_{i:02d}", float(v), step=step)
        for i, v in enumerate(cos_m.detach().cpu().tolist()):
            mlflow.log_metric(f"{suffix}_cossim_bucket_{i:02d}", float(v), step=step)
        for i, v in enumerate(log_m.detach().cpu().tolist()):
            mlflow.log_metric(
                f"{suffix}_log_norm_ratio_bucket_{i:02d}", float(v), step=step
            )
        mlflow.log_metric(f"{suffix}_mse_mean", float(mse_m.mean().item()), step=step)
        mlflow.log_metric(
            f"{suffix}_cossim_mean", float(cos_m.mean().item()), step=step
        )
        mlflow.log_metric(
            f"{suffix}_log_norm_ratio_mean", float(log_m.mean().item()), step=step
        )
        _log_bucket_histograms(
            stats, epoch=epoch, step=step, artifact_prefix=self.artifact_prefix
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(pl_module, "is_conditional", False):
            raise RuntimeError(
                "UnconditionalDebugCallback requires an unconditional model."
            )
        self._epoch_stats = BucketStats.zeros(self.num_buckets, pl_module.device)
        self._overall_stats = BucketStats.zeros(self.num_buckets, pl_module.device)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._epoch_stats = BucketStats.zeros(self.num_buckets, pl_module.device)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self._epoch_stats is None or self._overall_stats is None:
            raise RuntimeError(
                "UnconditionalDebugCallback was not initialised correctly."
            )

        images, _ = unpack_batch(batch)
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            mse_batch, t, v_est, v_true = _flow_matching_step_uncond(
                pl_module.generator,
                images,
                self._loss_fn,
                pl_module.device,
            )
            cossim_batch = F.cosine_similarity(
                v_est.flatten(1), v_true.flatten(1), dim=1
            )
            v_est_norm = torch.linalg.vector_norm(v_est.flatten(1), dim=1)
            v_true_norm = torch.linalg.vector_norm(v_true.flatten(1), dim=1)
            log_ratio_batch = torch.log(v_est_norm + 1e-6) - torch.log(
                v_true_norm + 1e-6
            )
            bins = torch.bucketize(
                t[:, 0, 0, 0], self._boundaries.to(pl_module.device), right=True
            )
            self._epoch_stats.update(mse_batch, cossim_batch, log_ratio_batch, bins)
            self._overall_stats.update(mse_batch, cossim_batch, log_ratio_batch, bins)

        if was_training:
            pl_module.train()

        if (
            self.every_n_steps > 0
            and (trainer.global_step + 1) % self.every_n_steps == 0
        ):
            self._maybe_log_summary(
                stats=self._epoch_stats,
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                suffix="debug",
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self._epoch_stats is None:
            raise RuntimeError(
                "UnconditionalDebugCallback was not initialised correctly."
            )
        self._maybe_log_summary(
            stats=self._epoch_stats,
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            suffix="debug_epoch",
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self._overall_stats is None:
            raise RuntimeError(
                "UnconditionalDebugCallback was not initialised correctly."
            )
        self._maybe_log_summary(
            stats=self._overall_stats,
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            suffix="debug_overall",
        )


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
    ) -> None:
        super().__init__(
            unet_cfg=unet_cfg,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters()

    def _build_generator(self) -> nn.Module:
        return SimpleColouriser.from_config(self.unet_cfg)

    def forward(self, LAB_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.generator(LAB_t, t)

    def _loss(self, batch) -> torch.Tensor:
        LAB, _ = unpack_batch(batch)
        LAB = LAB.to(self.device)
        L = LAB[:, :1, ...]
        ab = LAB[:, 1:, ...]
        B = ab.shape[0]
        x0 = torch.randn_like(ab)
        t = torch.rand(B, 1, 1, 1, device=ab.device, dtype=ab.dtype)
        ab_t = (1 - t) * x0 + t * ab
        LAB_t = torch.cat([L, ab_t], dim=1)
        v_est = self.generator(LAB_t, t)
        v_true = ab - x0
        return self.loss_fn(v_est, v_true)


def _get_sampling_metadata(trainer: pl.Trainer) -> tuple[tuple[int, ...], Any, Any]:
    datamodule = trainer.datamodule
    if datamodule is None:
        raise RuntimeError("Sampling callbacks require a datamodule with image_shape.")

    image_shape = getattr(datamodule, "image_shape", None)
    if image_shape is None:
        raise RuntimeError("Datamodule must expose image_shape for sampling callbacks.")

    sample_mean = getattr(datamodule, "sample_mean", None)
    sample_std = getattr(datamodule, "sample_std", None)
    return image_shape, sample_mean, sample_std


def _log_sample_image(trainer: pl.Trainer, image, artifact_prefix: str) -> None:
    logger = _get_mlflow_logger(trainer)
    with _mlflow_run_context(logger):
        mlflow.log_image(
            image,
            artifact_file=f"{artifact_prefix}/epoch_{trainer.current_epoch:04d}.png",
        )


class UnconditionalSampleCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 5,
        n_images: int = 64,
        nrow: int = 8,
        n_steps: int = 50,
        ode_solver_name: str = "euler_solver",
        sample_seed: int | None = 0,
        artifact_prefix: str = "samples",
    ) -> None:
        self.every_n_epochs = int(every_n_epochs)
        self.n_images = int(n_images)
        self.nrow = int(nrow)
        self.n_steps = int(n_steps)
        self.ode_solver_name = ode_solver_name
        self.sample_seed = sample_seed
        self.artifact_prefix = artifact_prefix

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        image_shape, sample_mean, sample_std = _get_sampling_metadata(trainer)
        ode_solver = get_ode_solver_from_name(self.ode_solver_name)

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            samples = sample_unconditional(
                model=pl_module.generator,
                n_images=self.n_images,
                image_shape=image_shape,
                ode_solver=ode_solver,
                n_steps=self.n_steps,
                return_all=False,
                device=pl_module.device,
                seed=self.sample_seed,
            )
            image = create_pil_image(
                samples,
                nrow=self.nrow,
                mean=sample_mean,
                std=sample_std,
            )

        if was_training:
            pl_module.train()

        _log_sample_image(trainer, image, self.artifact_prefix)


class ClassConditionalSampleCallback(Callback):
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

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        image_shape, sample_mean, sample_std = _get_sampling_metadata(trainer)
        ode_solver = get_ode_solver_from_name(self.ode_solver_name)

        num_classes = int(getattr(pl_module, "n_classes"))
        labels = torch.arange(num_classes, device=pl_module.device)
        repeats = max(1, -(-self.n_images // num_classes))
        labels = labels.repeat(repeats)[: self.n_images]

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            samples = sample_conditional(
                model=pl_module.generator,
                y=labels,
                image_shape=image_shape,
                ode_solver=ode_solver,
                n_steps=self.n_steps,
                guidance_scale=self.guidance_scale,
                null_id=getattr(pl_module, "null_id", None),
                return_all=False,
                device=pl_module.device,
                seed=self.sample_seed,
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

        _log_sample_image(trainer, image, self.artifact_prefix)


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

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("InputGridCallback requires a datamodule.")

        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        images, labels = unpack_batch(batch)

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
                images, _ = unpack_batch(batch)
                self._fid_metric.update(images, real=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
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
    def __init__(self, model_artifact_name: str, config_path: str) -> None:
        self.model_artifact_name = model_artifact_name
        self.config_path = config_path

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

        mlflow.log_artifact(self.config_path, artifact_path="configs")

        best_score = getattr(checkpoint_callback, "best_model_score", None)
        if best_score is not None:
            mlflow.log_metric("best_mse", float(best_score.detach().cpu().item()))

        mlflow.pytorch.log_model(
            getattr(pl_module, "generator"),
            artifact_path=self.model_artifact_name,
        )
