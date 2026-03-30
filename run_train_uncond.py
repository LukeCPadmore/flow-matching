import argparse
from datetime import datetime
import os
import shutil

import mlflow
import mlflow.pytorch
import torch

from models.config import OptimConfig, UNetConfig
from models.unet import UNet
from utils.create_dataloaders import create_mnist_train_val_loaders
from utils.logger_utils import get_temp_logger
from utils.train import create_pil_image, train_loop_uncond


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unconditional FM UNet on MNIST."
    )
    parser.add_argument(
        "--experiment-name", default="Flow Matching MNIST Unconditional"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--data-path",
        default="/home/luke-padmore/Source/flow-matching-mnist/data",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--mult", type=float, default=2.0)
    parser.add_argument("--d-trunk", type=int, default=8)
    parser.add_argument("--d-concat", type=int, default=8)
    parser.add_argument("--group-norm-size", type=int, default=8)
    parser.add_argument("--d-time", type=int, default=128)
    parser.add_argument("--max-time-period", type=float, default=10000.0)
    parser.add_argument(
        "--activation-name", default="silu", choices=["relu", "silu", "gelu"]
    )
    parser.add_argument(
        "--upsample-mode",
        default="nearest",
        choices=["nearest", "bilinear", "convtranspose"],
    )
    parser.add_argument(
        "--optim-name", default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sample-every-epochs", type=int, default=5)
    parser.add_argument("--sample-ode-steps", type=int, default=50)
    parser.add_argument("--sample-grid-size", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_cfg = UNetConfig(
        in_channels=1,
        base_channels=args.base_channels,
        mult=args.mult,
        n_layers=args.n_layers,
        d_trunk=args.d_trunk,
        d_concat=args.d_concat,
        group_norm_size=args.group_norm_size,
        d_time=args.d_time,
        max_time_period=args.max_time_period,
        activation_name=args.activation_name,
        upsample_mode=args.upsample_mode,
    )
    optim_cfg = OptimConfig(
        name=args.optim_name, lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader, val_loader = create_mnist_train_val_loaders(
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=args.num_workers,
        shuffle=True,
        transform="default",
    )
    images, _ = next(iter(train_loader))
    image_shape = tuple(images.shape[1:])

    model = UNet.from_config(unet_cfg).to(device)
    optim = optim_cfg.make_optimizer(model.parameters())

    mlflow.set_experiment(args.experiment_name)
    run_name = (
        args.run_name or f"train_uncond_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    logger, log_path = get_temp_logger("train_uncond")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(unet_cfg.to_mlflow_params(prefix="unet"))
        mlflow.log_params(optim_cfg.to_mlflow_params(prefix="optim"))
        mlflow.log_param("unet.channels", ",".join(map(str, unet_cfg.channels)))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("device", str(device))
        logger.info("Starting unconditional training run '%s'", run_name)
        logger.info("Device: %s", device)

        def on_step(global_step: int, mse_step: float, _epoch: int) -> None:
            if global_step % args.log_every_steps == 0:
                mlflow.log_metric("train_mse_step", float(mse_step), step=global_step)

        def on_epoch(epoch: int, train_mse: float, val_mse: float | None) -> None:
            mlflow.log_metric("train_mse_epoch", float(train_mse), step=epoch)
            if val_mse is not None:
                mlflow.log_metric("val_mse_epoch", float(val_mse), step=epoch)

        def on_sample(epoch: int, samples: torch.Tensor) -> None:
            img = create_pil_image(samples, nrow=args.sample_grid_size)
            mlflow.log_image(
                img,
                artifact_file=f"train_grids/uncond_samples_epoch_{epoch:04d}.png",
            )

        best = train_loop_uncond(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            optim=optim,
            device=device,
            on_step=on_step,
            on_epoch=on_epoch,
            sample_every_epochs=args.sample_every_epochs,
            sample_n_images=args.sample_grid_size * args.sample_grid_size,
            sample_image_shape=image_shape,
            sample_ode_steps=args.sample_ode_steps,
            sample_seed=args.sample_seed,
            on_sample=on_sample,
            logger=logger,
            log_every_steps=args.log_every_steps,
        )

        mlflow.log_metric("best_mse", float(best))
        mlflow.pytorch.log_model(model, name="UNet")
        logger.info("Finished unconditional training. best_mse=%.6f", float(best))
        mlflow.log_artifact(log_path, artifact_path="logs")
    shutil.rmtree(os.path.dirname(log_path), ignore_errors=True)


if __name__ == "__main__":
    main()
