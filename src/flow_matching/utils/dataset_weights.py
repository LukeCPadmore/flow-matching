from torch.utils.data import DataLoader
import torch


def compute_inv_pixel_weighting(dl: DataLoader, alpha=0.5, eps=1e-3, bins=32):
    points = []

    for batch, _ in dl:
        x = batch[:, 1:3]  # [B, 2, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = x.flatten(0, 2)  # [B*H*W, 2]
        points.append(x.cpu())

    points = torch.vstack(points)

    hist, bin_edges = torch.histogramdd(
        points,
        bins=(bins, bins),
        range=(-1.0, 1.0, -1.0, 1.0),
    )

    rarity = (hist + eps).pow(-alpha)

    return rarity, bin_edges


def compute_image_rarity(dl: DataLoader, rarity: torch.Tensor, bins):
    weights = []

    for batch, _ in dl:
        x = batch[:, 1:3]  # [B, 2, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = x.flatten(1, 2)  # [B, H*W, 2]

        a = x[..., 0].contiguous()  # [B, H*W]
        b = x[..., 1].contiguous()  # [B, H*W]

        a_bin = torch.bucketize(a, bins[0]) - 1
        b_bin = torch.bucketize(b, bins[1]) - 1

        a_bin = a_bin.clamp(0, rarity.shape[0] - 1)
        b_bin = b_bin.clamp(0, rarity.shape[1] - 1)

        pixel_rarity = rarity[a_bin, b_bin]  # [B, H*W]
        image_rarity = pixel_rarity.mean(dim=1)  # [B]

        weights.append(image_rarity.cpu())

    return torch.cat(weights)
