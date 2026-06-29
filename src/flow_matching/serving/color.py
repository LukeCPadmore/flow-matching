from __future__ import annotations

import numpy as np

from flow_matching.utils.color import lab_to_rgb_numpy, rgb_to_lab_numpy


def rgb_to_lab_bchw(rgb: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    return rgb_to_lab_numpy(rgb, normalize=normalize)


def lab_to_rgb_bchw(lab: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    return lab_to_rgb_numpy(lab, normalized=normalized)
