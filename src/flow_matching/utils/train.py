import torch
import torch.nn as nn
from PIL import ImageDraw
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from flow_matching.models.ode_solvers import (
    euler_solver,
    sample_conditional,
    sample_unconditional,
)


def _log(logger, message: str, *args) -> None:
    if logger is not None:
        logger.info(message, *args)
    else:
        print(message % args)


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Expected a non-empty batch.")
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return images, labels
    return batch, None


def flow_matching_step(model, x1, loss_fn, device):
    B = x1.shape[0]
    x1 = x1.to(device)
    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B, 1, 1, 1).to(device)
    v_est = model((1 - t) * x0 + x1 * t, t)
    v_true = x1 - x0
    mse = loss_fn(v_est, v_true)
    return mse


def flow_matching_step_cfg(model, x1, y, p_drop, null_id, loss_fn, device):
    B = x1.shape[0]
    x1 = x1.to(device)
    y = y.to(device)

    x0 = torch.randn_like(x1).to(device)
    t = torch.rand(B, 1, 1, 1).to(device)

    drop_mask = torch.rand_like(y.float()) < p_drop
    y_drop = y.clone()
    y_drop[drop_mask] = null_id

    v_est = model((1 - t) * x0 + x1 * t, t, y_drop)
    v_true = x1 - x0
    mse = loss_fn(v_est, v_true)
    return mse


def create_pil_image(
    images: torch.Tensor,
    nrow: int = 8,
    labels: torch.Tensor | list[int] | None = None,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
):
    images = images.detach().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
        images = images * std_t + mean_t
    elif images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0, 1)

    padding = 2
    grid = make_grid(images, nrow=nrow, padding=padding)
    img = to_pil_image(grid)
    if labels is not None:
        labels_list = (
            labels.detach().cpu().tolist()
            if isinstance(labels, torch.Tensor)
            else labels
        )
        draw = ImageDraw.Draw(img)
        tile_h = int(images.shape[-2])
        tile_w = int(images.shape[-1])
        text_fill = (255, 255, 255) if img.mode in ("RGB", "RGBA") else 255
        bg_fill = (0, 0, 0) if img.mode in ("RGB", "RGBA") else 0
        for idx, label in enumerate(labels_list):
            row = idx // nrow
            col = idx % nrow
            x = padding + col * (tile_w + padding) + 1
            y = padding + row * (tile_h + padding) + 1
            text = str(label)
            x1, y1, x2, y2 = draw.textbbox((x, y), text)
            draw.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), fill=bg_fill)
            draw.text((x, y), text, fill=text_fill)

    return img


def train_loop_uncond(
    model,
    train_loader,
    num_epochs: int,
    optim,
    device,
    val_loader=None,
    on_step=None,
    on_epoch=None,
    sample_every_epochs: int | None = None,
    sample_n_images: int | None = None,
    sample_image_shape=None,
    sample_ode_solver=euler_solver,
    sample_ode_steps: int = 50,
    sample_seed: int | None = 0,
    on_sample=None,
    logger=None,
    log_every_steps: int = 100,
):
    """
    on_step(global_step, train_mse_step, epoch)
    on_epoch(epoch, train_mse_epoch, val_mse_epoch)
        - val_mse_epoch is None if val_loader is None
    Returns:
        best_val_mse if val_loader is provided, else best_train_mse
    """
    loss_fn = nn.MSELoss()

    global_step = 0
    best_train = float("inf")
    best_val = float("inf")

    _log(
        logger,
        "train_loop_uncond: epochs=%d device=%s sample_every_epochs=%s sample_ode_steps=%d",
        num_epochs,
        str(device),
        str(sample_every_epochs),
        sample_ode_steps,
    )

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0

        for batch in train_loader:
            x1, _ = unpack_batch(batch)
            optim.zero_grad(set_to_none=True)
            mse = flow_matching_step(model, x1, loss_fn, device)
            mse.backward()
            optim.step()

            mse_step = float(mse.item())
            running += mse_step

            if on_step is not None:
                on_step(global_step, mse_step, epoch)
            if global_step % log_every_steps == 0:
                _log(
                    logger,
                    "[epoch %03d | step %06d] train_mse_step=%.6f",
                    epoch,
                    global_step,
                    mse_step,
                )

            global_step += 1

        train_mse_epoch = running / len(train_loader)
        best_train = min(best_train, train_mse_epoch)

        # val
        val_mse_epoch = None
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x1, _ = unpack_batch(batch)
                    mse = flow_matching_step(model, x1, loss_fn, device)
                    v_running += float(mse.item())
            val_mse_epoch = v_running / len(val_loader)
            best_val = min(best_val, val_mse_epoch)

        if on_epoch is not None:
            on_epoch(epoch, train_mse_epoch, val_mse_epoch)
        if val_mse_epoch is None:
            _log(
                logger,
                "[epoch %03d] train_mse_epoch=%.6f",
                epoch,
                train_mse_epoch,
            )
        else:
            _log(
                logger,
                "[epoch %03d] train_mse_epoch=%.6f val_mse_epoch=%.6f",
                epoch,
                train_mse_epoch,
                val_mse_epoch,
            )

        if sample_every_epochs is not None and sample_every_epochs > 0:
            should_sample = (epoch + 1) % sample_every_epochs == 0
            if should_sample:
                if sample_n_images is None or sample_image_shape is None:
                    raise ValueError(
                        "sample_n_images and sample_image_shape must be set when sample_every_epochs is enabled."
                    )
                samples = sample_unconditional(
                    model=model,
                    n_images=sample_n_images,
                    image_shape=sample_image_shape,
                    ode_solver=sample_ode_solver,
                    n_steps=sample_ode_steps,
                    return_all=False,
                    device=device,
                    seed=sample_seed,
                )
                if on_sample is not None:
                    on_sample(epoch, samples)
                _log(
                    logger,
                    "[epoch %03d] logged unconditional sample grid (n_images=%d, ode_steps=%d)",
                    epoch,
                    sample_n_images,
                    sample_ode_steps,
                )

    return best_val if val_loader is not None else best_train


def train_loop_class_cond(
    model,
    train_loader,
    num_epochs: int,
    optim,
    device,
    null_id: int,
    p_drop: float = 0.2,
    val_loader=None,
    on_step=None,
    on_epoch=None,
    sample_every_epochs=None,
    sample_n_rows=None,
    sample_classes=None,
    sample_image_shape=None,
    sample_ode_solver=euler_solver,
    sample_ode_steps=None,
    sample_guidance_scale=1.0,
    sample_seed=None,
    on_sample=None,
    logger=None,
    log_every_steps: int = 100,
):
    """
    on_step(global_step, train_mse_step, epoch)
    on_epoch(epoch, train_mse_epoch, val_mse_epoch)
        - val_mse_epoch is None if val_loader is None
    Returns:
        best_val_mse if val_loader is provided, else best_train_mse
    """
    loss_fn = nn.MSELoss()

    global_step = 0
    best_train = float("inf")
    best_val = float("inf")

    _log(
        logger,
        "train_loop_class_cond: epochs=%d device=%s p_drop=%.3f null_id=%d guidance_scale=%.3f",
        num_epochs,
        str(device),
        p_drop,
        null_id,
        sample_guidance_scale,
    )

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for batch in train_loader:
            x1, y = unpack_batch(batch)
            if y is None:
                raise ValueError(
                    "train_loop_class_cond requires labels. Set datamodule.drop_labels=False."
                )
            optim.zero_grad(set_to_none=True)
            mse = flow_matching_step_cfg(
                model=model,
                x1=x1,
                y=y,
                p_drop=p_drop,
                null_id=null_id,
                loss_fn=loss_fn,
                device=device,
            )
            mse.backward()
            optim.step()

            mse_step = float(mse.item())
            running += mse_step

            if on_step is not None:
                on_step(global_step, mse_step, epoch)
            if global_step % log_every_steps == 0:
                _log(
                    logger,
                    "[epoch %03d | step %06d] train_mse_step=%.6f",
                    epoch,
                    global_step,
                    mse_step,
                )
            global_step += 1

        train_mse_epoch = running / len(train_loader)
        best_train = min(best_train, train_mse_epoch)

        val_mse_epoch = None
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x1, y = unpack_batch(batch)
                    if y is None:
                        raise ValueError(
                            "train_loop_class_cond requires labels. Set datamodule.drop_labels=False."
                        )
                    mse = flow_matching_step_cfg(
                        model=model,
                        x1=x1,
                        y=y,
                        p_drop=0.0,
                        null_id=null_id,
                        loss_fn=loss_fn,
                        device=device,
                    )
                    v_running += float(mse.item())
            val_mse_epoch = v_running / len(val_loader)
            best_val = min(best_val, val_mse_epoch)

        if on_epoch is not None:
            on_epoch(epoch, train_mse_epoch, val_mse_epoch)
        if val_mse_epoch is None:
            _log(
                logger,
                "[epoch %03d] train_mse_epoch=%.6f",
                epoch,
                train_mse_epoch,
            )
        else:
            _log(
                logger,
                "[epoch %03d] train_mse_epoch=%.6f val_mse_epoch=%.6f",
                epoch,
                train_mse_epoch,
                val_mse_epoch,
            )

        if sample_every_epochs is not None and sample_every_epochs > 0:
            should_sample = (epoch + 1) % sample_every_epochs == 0
            if should_sample:
                if (
                    sample_classes is None
                    or sample_n_rows is None
                    or sample_image_shape is None
                ):
                    raise ValueError(
                        "sample_classes, sample_n_rows and sample_image_shape must be set when sample_every_epochs is enabled."
                    )
                sample_steps = 50 if sample_ode_steps is None else sample_ode_steps
                samples = sample_conditional(
                    model=model,
                    y=torch.arange(0, sample_classes).repeat(sample_n_rows),
                    image_shape=sample_image_shape,
                    ode_solver=sample_ode_solver,
                    n_steps=sample_steps,
                    guidance_scale=sample_guidance_scale,
                    null_id=null_id,
                    return_all=False,
                    device=device,
                    seed=sample_seed,
                )
                if on_sample is not None:
                    on_sample(epoch, samples)
                _log(
                    logger,
                    "[epoch %03d] logged conditional sample grid (classes=%d, rows=%d, ode_steps=%d, guidance_scale=%.3f)",
                    epoch,
                    sample_classes,
                    sample_n_rows,
                    sample_steps,
                    sample_guidance_scale,
                )

    return best_val if val_loader is not None else best_train
