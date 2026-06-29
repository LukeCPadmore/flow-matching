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
    return_all: bool = True,
):
    """Simple Euler integrator for numpy arrays."""
    h = (t1 - t0) / n_steps
    x = np.array(x0, copy=True)
    if return_all:
        xs, ts = [], []
        for k in range(n_steps + 1):
            t = t0 + k * h
            ts.append(t)
            xs.append(x.copy())
            if k < n_steps:
                x = x + h * f(x, t)
        return xs, ts

    for k in range(n_steps):
        t = t0 + k * h
        x = x + h * f(x, t)
    return x, None


def rk4_solver(
    f: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    return_all: bool = True,
):
    """Classical RK4 integrator for numpy arrays."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    h = (t1 - t0) / n_steps
    x = np.array(x0, copy=True)
    if return_all:
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

    for k in range(n_steps):
        t = t0 + k * h
        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, None


def sample_colouriser_ab(
    model,
    L,
    *,
    ode_solver,
    n_steps: int,
    guidance_scale: float = 1.0,
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
        if guidance_scale == 1.0:
            return model(ab, t, L)

        ab2 = np.concatenate([ab, ab], axis=0)
        t2 = np.concatenate([t, t], axis=0)
        L2 = np.concatenate([L, np.zeros_like(L)], axis=0)
        v2 = model(ab2, t2, L2)
        v_cond, v_uncond = np.split(v2, 2, axis=0)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    xs, _ = ode_solver(f, ab0, 0.0, 1.0, n_steps, return_all=return_all)

    if return_all:
        if clamp_mode == "clamp":
            xs = [np.clip(x, clamp_range[0], clamp_range[1]) for x in xs]
        elif clamp_mode == "tanh":
            xs = [np.tanh(x) for x in xs]
        elif clamp_mode is not None:
            raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
        return xs

    x = xs
    if clamp_mode == "clamp":
        x = np.clip(x, clamp_range[0], clamp_range[1])
    elif clamp_mode == "tanh":
        x = np.tanh(x)
    elif clamp_mode is not None:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
    return x
