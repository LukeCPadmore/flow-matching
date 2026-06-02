import argparse
from datetime import datetime
import os
import shutil

import mlflow
import mlflow.pytorch
import torch

from models.config import OptimConfig, UNetConfig
from models.unet import ClassCondUNet, UNet
from utils.create_dataloaders import create_cifar10_train_val_loaders
from utils.logger_utils import get_temp_logger
from utils.train import create_pil_image, train_loop_class_cond

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train class-conditional FM UNet on CIFAR-10."
    )
    parser.add_argument("--experiment-name", default="Flow Matching CIFAR10 Conditional")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--data-path",
        default="/home/luke-padmore/Source/flow-matching-mnist/data",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument(
        "--data-transform",
        default="default",
        choices=["default", "none"],
        help="CIFAR-10 transform preset",
    )

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--mult", type=float, default=2.0)
    parser.add_argument("--d-trunk", type=int, default=32)
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

    parser.add_argument("--d-cls-emb", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument(
        "--null-id",
        type=int,
        default=None,
        help="Defaults to n_classes when omitted",
    )
    parser.add_argument("--p-drop", type=float, default=0.2)

    parser.add_argument(
        "--optim-name", default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-every-epochs", type=int, default=5)
    parser.add_argument("--sample-ode-steps", type=int, default=50)
    parser.add_argument(
        "--sample-grid-nrows",
        type=int,
        default=10,
        help="Rows in conditional sample grid",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory where checkpoints are written before artifact upload",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=5,
        help="Save and log checkpoint every N epochs (set 0 to disable periodic saves)",
    )
    parser.add_argument(
        "--system-metrics-interval-s",
        type=int,
        default=10,
        help="MLflow system metrics sampling interval (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_cfg = UNetConfig(
        in_channels=3,
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

    train_loader, val_loader = create_cifar10_train_val_loaders(
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=args.num_workers,
        shuffle=True,
        transform=args.data_transform,
    )

    images, _ = next(iter(train_loader))
    image_shape = tuple(images.shape[1:])

    core = UNet.from_config(unet_cfg).to(device)
    null_id = args.null_id if args.null_id is not None else args.n_classes
    if null_id < 0:
        raise ValueError(f"null_id must be >= 0, got {null_id}")
    class_vocab_size = max(args.n_classes, null_id + 1)
    model = ClassCondUNet(
        core=core,
        n_classes=class_vocab_size,
        d_cls_emb=args.d_cls_emb,
    ).to(device)
    optim = optim_cfg.make_optimizer(model.parameters())

    mlflow.set_experiment(args.experiment_name)
    run_name = (
        args.run_name
        or f"train_cond_cifar10_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    logger, log_path = get_temp_logger("train_cond_cifar10")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(args.system_metrics_interval_s)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(unet_cfg.to_mlflow_params(prefix="unet"))
        mlflow.log_params(optim_cfg.to_mlflow_params(prefix="optim"))
        mlflow.log_param("dataset", "cifar10")
        mlflow.log_param("data_transform", args.data_transform)
        mlflow.log_param("unet.channels", ",".join(map(str, unet_cfg.channels)))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("device", str(device))
        mlflow.log_param("p_drop", args.p_drop)
        mlflow.log_param("n_classes", args.n_classes)
        mlflow.log_param("class_vocab_size", class_vocab_size)
        mlflow.log_param("null_id", null_id)
        mlflow.log_param("d_cls_emb", args.d_cls_emb)
        mlflow.log_param("checkpoint_dir", checkpoint_dir)
        mlflow.log_param("checkpoint_every_epochs", args.checkpoint_every_epochs)
        mlflow.log_param("system_metrics_interval_s", args.system_metrics_interval_s)
        logger.info("Starting class-conditional CIFAR-10 training run '%s'", run_name)
        logger.info("Device: %s", device)
        logger.info(
            "n_classes=%d class_vocab_size=%d null_id=%d",
            args.n_classes,
            class_vocab_size,
            null_id,
        )
        logger.info("Checkpoint dir: %s", checkpoint_dir)

        best_val_mse = float("inf")
        best_train_mse = float("inf")
        last_epoch = -1
        last_train_mse = float("nan")
        last_val_mse: float | None = None

        def save_checkpoint(
            filename: str,
            epoch: int,
            train_mse: float,
            val_mse: float | None,
        ) -> None:
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "train_mse_epoch": float(train_mse),
                    "val_mse_epoch": None if val_mse is None else float(val_mse),
                    "null_id": int(null_id),
                    "class_vocab_size": int(class_vocab_size),
                    "args": vars(args),
                },
                checkpoint_path,
            )
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        def on_step(global_step: int, mse_step: float, _epoch: int) -> None:
            if global_step % args.log_every_steps == 0:
                mlflow.log_metric("train_mse_step", float(mse_step), step=global_step)

        def on_epoch(epoch: int, train_mse: float, val_mse: float | None) -> None:
            nonlocal best_val_mse, best_train_mse, last_epoch, last_train_mse, last_val_mse
            last_epoch = int(epoch)
            last_train_mse = float(train_mse)
            last_val_mse = None if val_mse is None else float(val_mse)
            mlflow.log_metric("train_mse_epoch", float(train_mse), step=epoch)
            if val_mse is not None:
                mlflow.log_metric("val_mse_epoch", float(val_mse), step=epoch)
                if val_mse < best_val_mse:
                    best_val_mse = float(val_mse)
                    save_checkpoint("best.pt", epoch, train_mse, val_mse)
            elif train_mse < best_train_mse:
                best_train_mse = float(train_mse)
                save_checkpoint("best.pt", epoch, train_mse, val_mse)

            if args.checkpoint_every_epochs > 0 and (epoch + 1) % args.checkpoint_every_epochs == 0:
                save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, train_mse, val_mse)

        def on_sample(epoch: int, samples: torch.Tensor) -> None:
            labels = torch.arange(args.n_classes).repeat(args.sample_grid_nrows)
            img = create_pil_image(
                samples,
                nrow=args.sample_grid_nrows,
                labels=labels,
                mean=CIFAR10_MEAN if args.data_transform == "default" else None,
                std=CIFAR10_STD if args.data_transform == "default" else None,
            )
            mlflow.log_image(
                img,
                artifact_file=f"train_grids/cifar10_cond_samples_epoch_{epoch:04d}.png",
            )

        best = train_loop_class_cond(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            optim=optim,
            device=device,
            null_id=null_id,
            p_drop=args.p_drop,
            on_step=on_step,
            on_epoch=on_epoch,
            sample_every_epochs=args.sample_every_epochs,
            sample_n_rows=args.sample_grid_nrows,
            sample_classes=args.n_classes,
            sample_image_shape=image_shape,
            sample_ode_steps=args.sample_ode_steps,
            sample_guidance_scale=args.sample_guidance_scale,
            sample_seed=args.sample_seed,
            on_sample=on_sample,
            logger=logger,
            log_every_steps=args.log_every_steps,
        )

        mlflow.log_metric("best_mse", float(best))
        if last_epoch >= 0:
            save_checkpoint("final.pt", last_epoch, last_train_mse, last_val_mse)
        mlflow.pytorch.log_model(model, name="ClassCondUNet")
        logger.info(
            "Finished class-conditional CIFAR-10 training. best_mse=%.6f", float(best)
        )
        mlflow.log_artifact(log_path, artifact_path="logs")
    shutil.rmtree(os.path.dirname(log_path), ignore_errors=True)


if __name__ == "__main__":
    main()
