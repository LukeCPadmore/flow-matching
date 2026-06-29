import torch
from typing import Callable


@torch.no_grad()
def euler_solver(
    f: Callable,
    x0,
    t0: float,
    t1: float,
    n_steps: int,
    return_all: bool = True,
):
    """Simple Euler integrator."""
    # Split [0,1] into n_steps intervals
    h = (t1 - t0) / n_steps
    x = x0.clone()
    if return_all:
        xs, ts = [], []
        for k in range(n_steps + 1):
            t = t0 + k * h
            ts.append(t)
            xs.append(x.clone())
            if k < n_steps:
                dx = f(x, t)
                x = x + h * dx
        return xs, ts

    for k in range(n_steps):
        t = t0 + k * h
        dx = f(x, t)
        x = x + h * dx
    return x, None


@torch.no_grad()
def rk2_solver(
    f: Callable,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    n_steps: int,
    return_all: bool = True,
):
    """RK2 integrator."""
    # Split [0,1] into n_steps intervals
    h = (t1 - t0) / n_steps
    x = x0.clone()
    if return_all:
        xs, ts = [], []
        for k in range(n_steps + 1):
            t = t0 + k * h
            ts.append(t)
            xs.append(x.clone())
            if k < n_steps:
                k1 = f(x, t)
                x_pred = x + h * k1
                k2 = f(x_pred, t + h)
                x = x + 0.5 * h * (k1 + k2)
        return xs, ts

    for k in range(n_steps):
        t = t0 + k * h
        k1 = f(x, t)
        x_pred = x + h * k1
        k2 = f(x_pred, t + h)
        x = x + 0.5 * h * (k1 + k2)
    return x, None


@torch.no_grad()
def rk4_solver(
    f,
    x0,
    t0: float,
    t1: float,
    n_steps: int,
    return_all: bool = True,
):
    """Classical RK4 integrator."""
    assert n_steps >= 1, "n_steps must be >= 1"
    h = (t1 - t0) / n_steps
    x = x0.clone()
    if return_all:
        xs, ts = [], []
        for k in range(n_steps + 1):
            t = t0 + k * h
            ts.append(t)
            xs.append(x.clone())
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


def make_vf(
    model, *, y=None, guidance_scale: float | None = None, null_id: int | None = None
):
    model.eval()

    # unconditional
    if y is None:

        def f(x, t_scalar):
            t = torch.full(
                (x.size(0), 1, 1, 1), t_scalar, device=x.device, dtype=x.dtype
            )
            return model(x, t)

        return f

    # conditional no-guidance fast path
    w = 1.0 if guidance_scale is None else guidance_scale
    if w == 1.0:
        y_dev = y.to(next(model.parameters()).device)

        def f(x, t_scalar):
            t = torch.full(
                (x.size(0), 1, 1, 1), t_scalar, device=x.device, dtype=x.dtype
            )
            return model(x, t, y_dev)

        return f

    # CFG
    if null_id is None:
        raise ValueError("null_id is required when guidance_scale != 1")

    y_dev = y.to(next(model.parameters()).device)
    y_null = torch.full_like(y_dev, null_id)

    def f(x, t_scalar):
        B = x.size(0)
        t = torch.full((B, 1, 1, 1), t_scalar, device=x.device, dtype=x.dtype)
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        y2 = torch.cat([y_dev, y_null], dim=0)
        v2 = model(x2, t2, y2)
        v_cond, v_uncond = v2[:B], v2[B:]
        return v_uncond + w * (v_cond - v_uncond)

    return f


def create_samples(
    n_images: int,
    image_shape,  # (C, H, W)
    ode_solver,
    f,
    n_steps: int,
    return_all: bool = False,
    device=None,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",  # "clamp", "tanh", or None
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    """
    Samples n_images using the ODE solver.
    Returns:
      - if return_all=False: Tensor (B, C, H, W) at final time
      - if return_all=True:  list[Tensor] trajectory over time
    """

    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)

    x0 = torch.randn((n_images, *image_shape), device=device, generator=g)
    xs, _ = ode_solver(f, x0, 0.0, 1.0, n_steps, return_all=return_all)

    if return_all:
        if clamp_mode == "clamp":
            xs = [x.clamp_(*clamp_range) for x in xs]
        elif clamp_mode == "tanh":
            xs = [torch.tanh(x) for x in xs]
        elif clamp_mode is not None:
            raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
        return xs

    x = xs
    if clamp_mode == "clamp":
        x = x.clamp_(*clamp_range)
    elif clamp_mode == "tanh":
        x = torch.tanh(x)
    elif clamp_mode is not None:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
    return x


def sample_unconditional(
    model,
    n_images: int,
    image_shape,
    ode_solver,
    n_steps: int,
    return_all: bool = False,
    device=None,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    return create_samples(
        n_images=n_images,
        image_shape=image_shape,
        ode_solver=ode_solver,
        f=make_vf(model, y=None, guidance_scale=None, null_id=None),
        n_steps=n_steps,
        return_all=return_all,
        device=device,
        seed=seed,
        clamp_mode=clamp_mode,
        clamp_range=clamp_range,
    )


def sample_conditional(
    model,
    y,
    image_shape,
    ode_solver,
    n_steps: int,
    guidance_scale: float = 1.0,
    null_id: int | None = None,
    return_all: bool = False,
    device=None,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    return create_samples(
        n_images=y.size(0),
        image_shape=image_shape,
        ode_solver=ode_solver,
        f=make_vf(model, y=y, guidance_scale=guidance_scale, null_id=null_id),
        n_steps=n_steps,
        return_all=return_all,
        device=device,
        seed=seed,
        clamp_mode=clamp_mode,
        clamp_range=clamp_range,
    )


def sample_colouriser_ab(
    model,
    L,
    *,
    ode_solver,
    n_steps: int,
    guidance_scale: float = 1.0,
    return_all: bool = False,
    device=None,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    if L.ndim != 4 or L.shape[1] != 1:
        raise ValueError(f"Expected L to have shape [B, 1, H, W], got {tuple(L.shape)}")

    was_training = model.training
    model.eval()

    device = L.device if device is None else device
    L = L.to(device)
    batch_size, _, height, width = L.shape

    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)

    ab0 = torch.randn(
        (batch_size, 2, height, width),
        device=device,
        dtype=L.dtype,
        generator=g,
    )

    def f(ab, t_scalar: float):
        t = torch.full((batch_size, 1, 1, 1), t_scalar, device=device, dtype=L.dtype)
        if guidance_scale == 1.0:
            return model(ab, t, L)

        ab2 = torch.cat([ab, ab], dim=0)
        t2 = torch.cat([t, t], dim=0)
        L2 = torch.cat([L, torch.zeros_like(L)], dim=0)
        v2 = model(ab2, t2, L2)
        v_cond, v_uncond = v2[:batch_size], v2[batch_size:]
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    xs, _ = ode_solver(f, ab0, 0.0, 1.0, n_steps, return_all=return_all)

    if was_training:
        model.train()

    if return_all:
        if clamp_mode == "clamp":
            xs = [x.clamp_(*clamp_range) for x in xs]
        elif clamp_mode == "tanh":
            xs = [torch.tanh(x) for x in xs]
        elif clamp_mode is not None:
            raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
        return xs

    x = xs
    if clamp_mode == "clamp":
        x = x.clamp_(*clamp_range)
    elif clamp_mode == "tanh":
        x = torch.tanh(x)
    elif clamp_mode is not None:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}")
    return x


ODE_SOLVERS: dict[str, Callable] = {
    "euler_solver": euler_solver,
    "rk2_solver": rk2_solver,
    "rk4_solver": rk4_solver,
}


def get_ode_solver_from_name(name: str) -> Callable:
    if name not in ODE_SOLVERS:
        raise ValueError(
            f"Unknown ode_solver '{name}'. Expected one of {list(ODE_SOLVERS.keys())}"
        )
    return ODE_SOLVERS[name]
