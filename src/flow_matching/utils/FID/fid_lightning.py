from __future__ import annotations

from typing import Any, Sequence

import lightning.pytorch as pl
import mlflow
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from flow_matching.models.config import make_optimizer
from flow_matching.utils.mlflow_tracking_utils import load_lightning_checkpoint_path_from_run
from flow_matching.utils.train import unpack_batch


class FIDBackbone(nn.Module):
    def __init__(
        self,
        conv_block_channels: Sequence[int] = (1, 32, 64),
        fid_emb: int = 128,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.conv_block_channels = tuple(int(x) for x in conv_block_channels)
        self.fid_emb = int(fid_emb)
        self.num_features = self.fid_emb
        self.image_size = int(image_size)

        blocks = [
            ConvBlock(self.conv_block_channels[i], self.conv_block_channels[i + 1])
            for i in range(len(self.conv_block_channels) - 1)
        ]
        self.conv_blocks = nn.Sequential(*blocks)
        flattened_features = self.conv_block_channels[-1] * (
            self.image_size // (2 ** (len(self.conv_block_channels) - 1))
        ) ** 2
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, self.fid_emb),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        return self.embedding(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FIDClassifier(nn.Module):
    def __init__(
        self,
        conv_block_channels: Sequence[int] = (1, 32, 64),
        fid_emb: int = 128,
        n_classes: int = 10,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.conv_block_channels = tuple(int(x) for x in conv_block_channels)
        self.fid_emb = int(fid_emb)
        self.n_classes = int(n_classes)
        self.image_size = int(image_size)
        self.backbone = FIDBackbone(
            conv_block_channels=self.conv_block_channels,
            fid_emb=self.fid_emb,
            image_size=self.image_size,
        )
        self.head = nn.Linear(self.backbone.num_features, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


FID_classifier = FIDClassifier


class FIDClassifierLightningModule(pl.LightningModule):
    def __init__(
        self,
        classifier_cfg: dict[str, Any] | None = None,
        optim_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.classifier_cfg = dict(classifier_cfg or {})
        self.optim_cfg = dict(optim_cfg or {})
        self.model = FIDClassifier(**self.classifier_cfg)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = unpack_batch(batch)
        if y is None:
            raise ValueError(
                "FIDClassifierLightningModule requires labels. Set datamodule.drop_labels=False."
            )
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return logits, y, loss

    def training_step(self, batch, batch_idx):
        logits, y, loss = self._shared_step(batch)
        batch_size = int(y.shape[0])
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, y, loss = self._shared_step(batch)
        batch_size = int(y.shape[0])
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return make_optimizer(
            self.parameters(),
            optimizer_name=str(self.optim_cfg.get("name", "adamw")),
            lr=float(self.optim_cfg.get("lr", 3e-4)),
            weight_decay=float(self.optim_cfg.get("weight_decay", 1e-4)),
        )

    @classmethod
    def load_backbone_from_mlflow_run(
        cls,
        run_id: str,
        *,
        artifact_path: str = "checkpoints/best.ckpt",
        map_location: str | torch.device = "cpu",
    ) -> nn.Module:
        checkpoint_path = load_lightning_checkpoint_path_from_run(
            run_id,
            artifact_path=artifact_path,
        )
        module = cls.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
        )
        backbone = module.backbone
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        return backbone


class FIDCheckpointArtifactCallback(Callback):
    def __init__(self, artifact_path: str = "checkpoints") -> None:
        self.artifact_path = artifact_path

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        if best_model_path:
            mlflow.log_artifact(best_model_path, artifact_path=self.artifact_path)

        last_model_path = getattr(checkpoint_callback, "last_model_path", "")
        if last_model_path:
            mlflow.log_artifact(last_model_path, artifact_path=self.artifact_path)
