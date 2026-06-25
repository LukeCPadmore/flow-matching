from __future__ import annotations

import numpy as np
import torch
from skimage.color import lab2rgb as _lab2rgb
from skimage.color import rgb2lab as _rgb2lab


def _ensure_numpy_bchw(
    x: np.ndarray, channels: int, name: str
) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float32)
    single = x.ndim == 3
    if single:
        x = x[None, ...]
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(
            f"Expected {name} to have shape [B, {channels}, H, W], got {x.shape}"
        )
    return x, single


def _ensure_torch_bchw(
    x: torch.Tensor, channels: int, name: str
) -> tuple[torch.Tensor, bool]:
    single = x.ndim == 3
    if single:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(
            f"Expected {name} to have shape [B, {channels}, H, W], got {tuple(x.shape)}"
        )
    return x, single


def rgb_to_lab_numpy(rgb: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    rgb, single = _ensure_numpy_bchw(rgb, 3, "rgb")
    lab = np.stack(
        [_rgb2lab(np.moveaxis(image, 0, -1)) for image in rgb],
        axis=0,
    ).astype(np.float32)
    lab = np.moveaxis(lab, -1, 1)

    if normalize:
        lab = lab.copy()
        lab[:, :1] /= 100.0
        lab[:, 1:] /= 128.0

    return lab[0] if single else lab


def lab_to_rgb_numpy(lab: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    lab, single = _ensure_numpy_bchw(lab, 3, "lab")
    lab = lab.astype(np.float32, copy=True)

    if normalized:
        lab[:, :1] *= 100.0
        lab[:, 1:] *= 128.0

    rgb = np.stack(
        [_lab2rgb(np.moveaxis(image, 0, -1)) for image in lab],
        axis=0,
    ).astype(np.float32)
    rgb = np.moveaxis(rgb, -1, 1).clip(0.0, 1.0)

    return rgb[0] if single else rgb


def rgb_to_lab_torch(rgb: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    device = rgb.device
    dtype = rgb.dtype if torch.is_floating_point(rgb) else torch.float32
    rgb, single = _ensure_torch_bchw(rgb, 3, "rgb")
    lab = rgb_to_lab_numpy(
        rgb.detach().to(torch.float32).cpu().numpy(), normalize=normalize
    )
    lab = torch.from_numpy(np.asarray(lab))
    lab = lab.to(device=device, dtype=dtype)
    return lab[0] if single else lab


def lab_to_rgb_torch(lab: torch.Tensor, *, normalized: bool = True) -> torch.Tensor:
    device = lab.device
    dtype = lab.dtype if torch.is_floating_point(lab) else torch.float32
    lab, single = _ensure_torch_bchw(lab, 3, "lab")
    rgb = lab_to_rgb_numpy(
        lab.detach().to(torch.float32).cpu().numpy(), normalized=normalized
    )
    rgb = torch.from_numpy(np.asarray(rgb))
    rgb = rgb.to(device=device, dtype=dtype)
    return rgb[0] if single else rgb


class RGBToLAB:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return rgb_to_lab_torch(img)
