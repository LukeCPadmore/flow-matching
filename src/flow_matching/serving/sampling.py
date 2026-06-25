from __future__ import annotations

from typing import Callable

import numpy as np

# Numpy copies of ODE solvers to make serving container more lightweight.


def euler_solver(
    f: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
):
    """Simple Euler integrator for numpy arrays."""
    h = (t1 - t0) / n_steps
    x = np.array(x0, copy=True)
    xs, ts = [], []
    for k in range(n_steps + 1):
        t = t0 + k * h
        ts.append(t)
        xs.append(x.copy())
        if k < n_steps:
            x = x + h * f(x, t)
    return xs, ts


def rk4_solver(
    f: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
):
    """Classical RK4 integrator for numpy arrays."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    h = (t1 - t0) / n_steps
    x = np.array(x0, copy=True)
    xs, ts = [], []
    for k in range(n_steps + 1):
        t = t0 + k * h
        ts.append(t)
        xs.append(x.copy())
        if k < n_steps:
            k1 = f(x, t)
            k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = f(x + h * k3, t + h)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xs, ts


def sample_colouriser_ab(
    model,
    L,
    *,
    ode_solver,
    n_steps: int,
    return_all: bool = False,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    """Sample AB channels conditioned on a normalized L channel batch."""
    L = np.asarray(L, dtype=np.float32)
    if L.ndim != 4 or L.shape[1] != 1:
        raise ValueError(f"Expected L to have shape [B, 1, H, W], got {L.shape}")

    batch_size, _, height, width = L.shape

    rng = np.random.default_rng(seed)
    ab0 = rng.standard_normal((batch_size, 2, height, width), dtype=np.float32)

    def f(ab: np.ndarray, t_scalar: float) -> np.ndarray:
        t = np.full((batch_size, 1, 1, 1), t_scalar, dtype=np.float32)
        LAB_t = np.concatenate([L, ab], axis=1)
        return model(LAB_t, t)

    xs, _ = ode_solver(f, ab0, 0.0, 1.0, n_steps)

    if clamp_mode == "clamp":
        xs = [np.clip(x, clamp_range[0], clamp_range[1]) for x in xs]
    elif clamp_mode == "tanh":
        xs = [np.tanh(x) for x in xs]
    elif clamp_mode is not None:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}")

    return xs if return_all else xs[-1]
