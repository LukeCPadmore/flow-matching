from __future__ import annotations

from contextlib import contextmanager
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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import WeightAveraging
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.swa_utils import get_ema_avg_fn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flow_matching.models.config import UNetConfig, make_optimizer
from flow_matching.models.ode_solvers import (
    get_ode_solver_from_name,
    sample_colouriser_ab,
    sample_conditional,
    sample_unconditional,
)
from flow_matching.models.unet import (
    ClassCondUNet,
    LEncoderColouriser,
    SimpleColouriser,
    UNet,
)
from flow_matching.utils.FID.fid_evaluation import build_fid_metric
from flow_matching.utils.FID.fid_lightning import FIDClassifierLightningModule
from flow_matching.utils.data_modules import lab_to_rgb
from flow_matching.utils.train import (
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


@dataclass
class RunningABStats:
    sum: torch.Tensor
    sq_sum: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor
    count: int
    true_ab_hist2d: torch.Tensor | None
    pred_ab_hist2d: torch.Tensor | None
    hist_bins: int
    hist_low: float
    hist_high: float

    @classmethod
    def zeros(
        cls,
        device: torch.device,
        *,
        hist_bins: int = 0,
        hist_low: float = -1.0,
        hist_high: float = 1.0,
    ) -> "RunningABStats":
        z = torch.zeros(2, device=device, dtype=torch.float64)
        true_ab_hist2d = None
        pred_ab_hist2d = None
        if hist_bins > 0:
            true_ab_hist2d = torch.zeros((hist_bins, hist_bins), dtype=torch.float64)
            pred_ab_hist2d = torch.zeros((hist_bins, hist_bins), dtype=torch.float64)
        return cls(
            sum=z.clone(),
            sq_sum=z.clone(),
            min=torch.full((2,), float("inf"), device=device, dtype=torch.float64),
            max=torch.full((2,), float("-inf"), device=device, dtype=torch.float64),
            count=0,
            true_ab_hist2d=true_ab_hist2d,
            pred_ab_hist2d=pred_ab_hist2d,
            hist_bins=hist_bins,
            hist_low=hist_low,
            hist_high=hist_high,
        )

    def update(self, ab: torch.Tensor, paired_ab: torch.Tensor | None = None) -> None:
        ab = ab.detach().to(dtype=torch.float64)
        self.sum += ab.sum(dim=(0, 2, 3))
        self.sq_sum += (ab * ab).sum(dim=(0, 2, 3))
        self.min = torch.minimum(self.min, ab.amin(dim=(0, 2, 3)))
        self.max = torch.maximum(self.max, ab.amax(dim=(0, 2, 3)))
        self.count += int(ab.shape[0] * ab.shape[2] * ab.shape[3])
        if paired_ab is not None and self.true_ab_hist2d is not None:
            self._update_ab_histogram(paired_ab, self.true_ab_hist2d)
            self._update_ab_histogram(ab, self.pred_ab_hist2d)

    def _update_ab_histogram(
        self, ab: torch.Tensor, hist2d: torch.Tensor | None
    ) -> None:
        if hist2d is None:
            return
        ab = ab.detach().to(dtype=torch.float64, device="cpu")
        scale = (self.hist_bins - 1) / max(self.hist_high - self.hist_low, 1e-12)
        a_idx = torch.clamp(
            ((ab[:, 0] - self.hist_low) * scale).floor().to(torch.long),
            0,
            self.hist_bins - 1,
        )
        b_idx = torch.clamp(
            ((ab[:, 1] - self.hist_low) * scale).floor().to(torch.long),
            0,
            self.hist_bins - 1,
        )
        bins = (b_idx * self.hist_bins + a_idx).reshape(-1)
        counts = torch.bincount(bins, minlength=self.hist_bins * self.hist_bins).to(
            dtype=torch.float64
        )
        hist2d += counts.view(self.hist_bins, self.hist_bins)

    def mean(self) -> torch.Tensor:
        safe_count = max(self.count, 1)
        return self.sum / safe_count

    def std(self) -> torch.Tensor:
        safe_count = max(self.count, 1)
        mean = self.mean()
        var = self.sq_sum / safe_count - mean.square()
        return var.clamp_min(0.0).sqrt()

    def overall_mean(self) -> torch.Tensor:
        safe_count = max(self.count * 2, 1)
        return self.sum.sum() / safe_count

    def overall_std(self) -> torch.Tensor:
        safe_count = max(self.count * 2, 1)
        overall_mean = self.overall_mean()
        var = self.sq_sum.sum() / safe_count - overall_mean.square()
        return var.clamp_min(0.0).sqrt()

    def overall_min(self) -> torch.Tensor:
        return self.min.min()

    def overall_max(self) -> torch.Tensor:
        return self.max.max()

    def to_metrics(self, prefix: str) -> dict[str, torch.Tensor]:
        mean = self.mean().to(dtype=torch.float32)
        std = self.std().to(dtype=torch.float32)
        return {
            f"{prefix}_ab_min": self.overall_min().to(dtype=torch.float32),
            f"{prefix}_ab_max": self.overall_max().to(dtype=torch.float32),
            f"{prefix}_ab_mean": self.overall_mean().to(dtype=torch.float32),
            f"{prefix}_ab_std": self.overall_std().to(dtype=torch.float32),
            f"{prefix}_a_mean": mean[0],
            f"{prefix}_a_std": std[0],
            f"{prefix}_b_mean": mean[1],
            f"{prefix}_b_std": std[1],
        }


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


def _log_colouriser_ab_histogram(
    trainer: pl.Trainer, stats: RunningABStats, artifact_prefix: str
) -> None:
    if stats.true_ab_hist2d is None or stats.pred_ab_hist2d is None:
        return

    true_hist = stats.true_ab_hist2d.detach().cpu().to(dtype=torch.float32)
    pred_hist = stats.pred_ab_hist2d.detach().cpu().to(dtype=torch.float32)
    hist_bins = int(stats.hist_bins)
    edges = np.linspace(stats.hist_low, stats.hist_high, hist_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0]) if hist_bins > 0 else 1.0
    extent = [stats.hist_low, stats.hist_high, stats.hist_low, stats.hist_high]
    vmax = float(max(true_hist.max().item(), pred_hist.max().item(), 1.0))

    fig = plt.figure(figsize=(15, 6))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.35)

    image = None
    panels = (
        (outer[0, 0], true_hist, "true a vs b"),
        (outer[0, 1], pred_hist, "pred a vs b"),
    )
    for outer_cell, hist2d, title in panels:
        block = outer_cell.subgridspec(
            2,
            2,
            height_ratios=[1.0, 4.0],
            width_ratios=[4.0, 1.0],
            hspace=0.05,
            wspace=0.05,
        )
        ax_top = fig.add_subplot(block[0, 0])
        ax_joint = fig.add_subplot(block[1, 0], sharex=ax_top)
        ax_side = fig.add_subplot(block[1, 1], sharey=ax_joint)

        a_marginal = hist2d.sum(dim=0).numpy()
        b_marginal = hist2d.sum(dim=1).numpy()

        ax_top.bar(
            centers,
            a_marginal,
            width=bin_width,
            align="center",
            color="#dddddd",
            edgecolor="none",
        )
        ax_top.set_xlim(stats.hist_low, stats.hist_high)
        ax_top.set_ylabel("count")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_title(title)

        image = ax_joint.imshow(
            hist2d.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax_joint.plot(
            [stats.hist_low, stats.hist_high],
            [stats.hist_low, stats.hist_high],
            color="white",
            linestyle="--",
            linewidth=1,
        )
        ax_joint.set_xlabel("a")
        ax_joint.set_ylabel("b")

        ax_side.barh(
            centers,
            b_marginal,
            height=bin_width,
            align="center",
            color="#dddddd",
            edgecolor="none",
        )
        ax_side.set_ylim(stats.hist_low, stats.hist_high)
        ax_side.tick_params(axis="y", labelleft=False)
        ax_side.set_xlabel("count")

    cax = fig.add_subplot(outer[0, 2])
    if image is not None:
        fig.colorbar(image, cax=cax, label="count")

    fig.suptitle(f"epoch {trainer.current_epoch:03d} colouriser ab distribution")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
    with _mlflow_run_context(trainer):
        mlflow.log_image(
            img,
            artifact_file=f"{artifact_prefix}/epoch_{trainer.current_epoch:04d}.png",
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
                torch.rand((B, 1, 1, 1), device=ab.device, dtype=ab.dtype)
                < self.p_drop
            )
            L = torch.where(drop_mask, torch.zeros_like(L), L)
        v_est = self.generator(ab_t, t, L)
        v_true = ab - x0
        return self.loss_fn(v_est, v_true)


class LEncoderColouriserFlowMatchingModule(SimpleColouriserFlowMatchingModule):
    model_artifact_name = "LEncoderColouriser"

    def _build_generator(self) -> nn.Module:
        return LEncoderColouriser.from_config(self.unet_cfg)


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


def _get_mlflow_logger(trainer: pl.Trainer) -> MLFlowLogger:
    logger = trainer.logger
    if isinstance(logger, MLFlowLogger):
        return logger

    if isinstance(logger, (list, tuple)):
        for item in logger:
            if isinstance(item, MLFlowLogger):
                return item

    raise RuntimeError("Expected trainer to be configured with an MLFlowLogger.")


def _get_model_checkpoint_callback(trainer: pl.Trainer) -> ModelCheckpoint:
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    raise RuntimeError("Expected trainer to be configured with a ModelCheckpoint.")


@contextmanager
def _mlflow_run_context(trainer: pl.Trainer):
    logger = _get_mlflow_logger(trainer)
    active = mlflow.active_run()
    if active is not None and active.info.run_id == logger.run_id:
        yield
        return

    with mlflow.start_run(run_id=logger.run_id):
        yield


def _log_sample_image(trainer: pl.Trainer, image, artifact_prefix: str) -> None:
    with _mlflow_run_context(trainer):
        mlflow.log_image(
            image,
            artifact_file=f"{artifact_prefix}/epoch_{trainer.current_epoch:04d}.png",
        )


class EMAWeightAveraging(WeightAveraging):
    def __init__(
        self,
        start_step: int = 100,
        decay: float = 0.999,
        device: str | torch.device | int | None = None,
        use_buffers: bool = False,
    ) -> None:
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.start_step = int(start_step)
        self.decay = float(decay)
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            avg_fn=get_ema_avg_fn(decay=self.decay),
        )

    def should_update(
        self, step_idx: int | None = None, epoch_idx: int | None = None
    ) -> bool:
        return step_idx is not None and step_idx >= self.start_step


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


class ColouriserLPIPSCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 5,
        n_images: int = 64,
        nrow: int = 8,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        ode_solver_name: str = "euler_solver",
        clamp_mode: str = "clamp",
        sample_seed: int | None = 0,
    ) -> None:
        self.every_n_epochs = int(every_n_epochs)
        self.n_images = int(n_images)
        self.nrow = int(nrow)
        self.n_steps = int(n_steps)
        self.guidance_scale = float(guidance_scale)
        self.ode_solver_name = ode_solver_name
        self.sample_seed = sample_seed
        self.clamp_mode = clamp_mode
        self._lpips_metric: LearnedPerceptualImagePatchSimilarity | None = None
        self._target_stats_logged = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.datamodule is None:
            raise RuntimeError("ColouriserLPIPSCallback requires a datamodule.")

        try:
            self._lpips_metric = LearnedPerceptualImagePatchSimilarity("vgg").to(
                pl_module.device
            )
        except Exception as exc:
            raise RuntimeError(
                "ColouriserLPIPSCallback requires LPIPS vgg weights to be available in the environment."
            ) from exc

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if self.every_n_epochs <= 0:
            return

        if self._lpips_metric is None:
            raise RuntimeError("ColouriserLPIPSCallback was not initialised correctly.")

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("ColouriserLPIPSCallback requires a datamodule.")

        ode_solver = get_ode_solver_from_name(self.ode_solver_name)
        loader = datamodule.val_dataloader()

        self._lpips_metric = self._lpips_metric.to(pl_module.device)
        self._lpips_metric.reset()

        sample_count = 0
        ab_mse_sum = 0.0
        rgb_mse_sum = 0.0
        pred_ab_stats = RunningABStats.zeros(pl_module.device, hist_bins=64)
        target_ab_stats = RunningABStats.zeros(pl_module.device)
        comparison_tiles: list[torch.Tensor] = []
        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            for batch in loader:
                LAB, _ = unpack_batch(batch)
                LAB = LAB.to(pl_module.device)

                take = min(LAB.shape[0], self.n_images - sample_count)
                if take <= 0:
                    break

                LAB = LAB[:take]
                L = LAB[:, :1, ...]
                ab = LAB[:, 1:, ...]

                ab_pred = sample_colouriser_ab(
                    pl_module.generator,
                    L,
                    n_steps=self.n_steps,
                    guidance_scale=self.guidance_scale,
                    ode_solver=ode_solver,
                    seed=self.sample_seed,
                    device=pl_module.device,
                    clamp_mode=self.clamp_mode,
                )
                pred_ab_stats.update(ab_pred, paired_ab=ab)
                target_ab_stats.update(ab)

                pred_lab = torch.cat([L, ab_pred.clamp(-1.0, 1.0)], dim=1)

                gt_rgb = lab_to_rgb(LAB)
                pred_rgb = lab_to_rgb(pred_lab)
                self._lpips_metric.update(
                    pred_rgb.mul(2.0).sub(1.0), gt_rgb.mul(2.0).sub(1.0)
                )
                comparison_tiles.append(
                    torch.cat([L.repeat(1, 3, 1, 1), pred_rgb, gt_rgb], dim=-1)
                )

                batch_size = int(LAB.shape[0])
                ab_mse_sum += float(F.mse_loss(ab_pred, ab).item()) * batch_size
                rgb_mse_sum += float(F.mse_loss(pred_rgb, gt_rgb).item()) * batch_size
                sample_count += batch_size

        if was_training:
            pl_module.train()

        if sample_count <= 0:
            raise RuntimeError("ColouriserLPIPSCallback found no validation samples.")

        lpips = self._lpips_metric.compute()
        ab_mse = torch.tensor(
            ab_mse_sum / sample_count, device=pl_module.device, dtype=torch.float32
        )
        rgb_mse = torch.tensor(
            rgb_mse_sum / sample_count, device=pl_module.device, dtype=torch.float32
        )

        pred_metrics = pred_ab_stats.to_metrics("val_pred")
        target_metrics = target_ab_stats.to_metrics("val_target")

        _log_colouriser_ab_histogram(trainer, pred_ab_stats, "colouriser/ab_hist")

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0 and comparison_tiles:
            comparison_image = create_pil_image(
                torch.cat(comparison_tiles, dim=0),
                nrow=self.nrow,
            )
            _log_sample_image(trainer, comparison_image, "colouriser/comparisons")

        pl_module.log(
            "val_lpips",
            lpips,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sample_count,
            sync_dist=False,
        )
        pl_module.log(
            "val_rgb_mse",
            rgb_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=sample_count,
            sync_dist=False,
        )
        pl_module.log(
            "val_ab_mse",
            ab_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=sample_count,
            sync_dist=False,
        )

        for name, value in pred_metrics.items():
            pl_module.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=sample_count,
                sync_dist=False,
            )

        if not self._target_stats_logged:
            for name, value in target_metrics.items():
                pl_module.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=sample_count,
                    sync_dist=False,
                )
            self._target_stats_logged = True

        self._lpips_metric.reset()


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
        with _mlflow_run_context(trainer):
            mlflow.log_image(
                image, artifact_file=f"{self.artifact_prefix}/train_start.png"
            )


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

        with _mlflow_run_context(trainer):
            mlflow.log_metric("fid", fid, step=trainer.current_epoch)


class MLFlowArtifactCallback(Callback):
    def __init__(
        self,
        model_artifact_name: str,
        config_path: str,
        best_score_metric_name: str = "best_mse",
    ) -> None:
        self.model_artifact_name = model_artifact_name
        self.config_path = config_path
        self.best_score_metric_name = best_score_metric_name

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        checkpoint_callback = _get_model_checkpoint_callback(trainer)
        with _mlflow_run_context(trainer):
            best_model_path = getattr(checkpoint_callback, "best_model_path", "")
            if best_model_path:
                mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

            last_model_path = getattr(checkpoint_callback, "last_model_path", "")
            if last_model_path:
                mlflow.log_artifact(last_model_path, artifact_path="checkpoints")

            mlflow.log_artifact(self.config_path, artifact_path="configs")

            best_score = getattr(checkpoint_callback, "best_model_score", None)
            if best_score is not None:
                mlflow.log_metric(
                    self.best_score_metric_name,
                    float(best_score.detach().cpu().item()),
                )

            mlflow.pytorch.log_model(
                getattr(pl_module, "generator"),
                name=self.model_artifact_name,
            )
