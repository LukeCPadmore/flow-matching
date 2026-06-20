from __future__ import annotations

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.FID.fid_lightning import FIDClassifierLightningModule


def build_fid_metric(backbone: torch.nn.Module) -> FrechetInceptionDistance:
    metric = FrechetInceptionDistance(
        feature=backbone,
        reset_real_features=False,
        normalize=False,
    )
    return metric


@torch.no_grad()
def _update_metric_from_loader(
    metric: FrechetInceptionDistance,
    dataloader: DataLoader,
    *,
    real: bool,
    device,
) -> None:
    for batch in dataloader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        metric.update(x.to(device), real=real)


@torch.no_grad()
def evaluate_fid_with_lightning_backbone(
    *,
    sample_fn,
    device,
    backbone_run_id: str,
    backbone_artifact_path: str = "checkpoints/best.ckpt",
    n_samples: int = 5210,
    batch_size: int = 128,
    real_loader: DataLoader | None = None,
    show_progress: bool = True,
):
    if real_loader is None:
        raise ValueError("real_loader is required when using TorchMetrics FID.")

    backbone = FIDClassifierLightningModule.load_backbone_from_mlflow_run(
        backbone_run_id,
        artifact_path=backbone_artifact_path,
        map_location=device,
    ).to(device).eval()
    metric = build_fid_metric(backbone).to(device)

    _update_metric_from_loader(metric, real_loader, real=True, device=device)

    gen_embs = []
    remaining = n_samples
    pbar = tqdm(total=n_samples, desc="Generating embeddings", disable=not show_progress)
    while remaining > 0:
        b = min(batch_size, remaining)
        imgs = sample_fn(b)
        if not torch.is_tensor(imgs):
            raise TypeError("sample_fn must return a torch.Tensor")

        imgs = imgs.to(device)
        z = backbone(imgs)
        z = z.view(z.size(0), -1)
        gen_embs.append(z.detach().cpu().numpy())
        metric.update(imgs, real=False)
        remaining -= b
        pbar.update(b)
    pbar.close()

    gen_embs = np.concatenate(gen_embs, axis=0)
    fid = float(metric.compute().detach().cpu().item())
    metric.reset()

    return fid, gen_embs, None
