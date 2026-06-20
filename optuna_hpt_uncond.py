from pathlib import Path

import mlflow
import optuna
import typer
import yaml

from models.config import log_config_kv, make_optimizer
from models.unet import UNet
from utils.data_modules import MNISTDataModule
from utils.logger_utils import trial_logger
from utils.optuna_models import HPTYaml
from utils.train import train_loop_uncond

app = typer.Typer(no_args_is_help=True)


def load_hpt_config(config_path: Path) -> HPTYaml:
    data = yaml.safe_load(config_path.read_text())
    return HPTYaml.model_validate(data)


def create_study_from_cfg(hpt: HPTYaml) -> optuna.Study:
    # Only pass keys Optuna expects
    kwargs = hpt.opt_study_cfg.model_dump(
        include={"study_name", "direction", "storage"}
    )
    return optuna.create_study(load_if_exists=True, **kwargs)


def make_objective(
    train_loader,
    hpt: HPTYaml,
    val_loader=None,
    experiment_name: str = "optuna_hpt",
    device: str = "cuda",
    num_epochs: int = 3,
    log_every_steps: int = 10,
):
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        unet_cfg, optim_params = hpt.sample(trial)
        model = UNet.from_config(unet_cfg).to(device)
        optim = make_optimizer(model.parameters(), **optim_params)

        run_name = f"trial_{trial.number:04d}"
        with (
            mlflow.start_run(run_name=run_name),
            trial_logger(run_name) as logger,
        ):
            logger.info(f"Starting trial {trial.number}")
            log_config_kv(unet_cfg, logger, prefix="unet")
            log_config_kv(optim_params, logger, prefix="optim")
            # useful tags
            mlflow.set_tag("optuna.trial_number", trial.number)
            mlflow.set_tag("study_name", experiment_name)

            # log params
            mlflow.log_params(unet_cfg.to_mlflow_params(prefix="unet"))
            mlflow.log_params({f"optim.{k}": v for k, v in optim_params.items()})

            def on_epoch(epoch, train_mse, val_mse):
                logger.info(f"[epoch {epoch}] train_mse={train_mse:.6f}")
                mlflow.log_metric("train_mse_epoch", float(train_mse), step=epoch)
                if val_mse is not None:
                    mlflow.log_metric("val_mse_epoch", float(val_mse), step=epoch)

                # pruning uses the objective signal (choose train or val)
                score = float(val_mse) if val_mse is not None else float(train_mse)
                trial.report(score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            def on_step(global_step, mse_step, epoch):
                if global_step % log_every_steps == 0:
                    logger.info(
                        f"[epoch {epoch} | global_step {global_step:06d}] train_mse={mse_step:.6f}"
                    )
                    mlflow.log_metric(
                        "train_mse_step", float(mse_step), step=global_step
                    )

            best = train_loop_uncond(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                optim=optim,
                device=device,
                on_epoch=on_epoch,
                on_step=on_step,
            )

            mlflow.log_metric("best_mse", float(best))
            logger.info(f"Finished trial {trial.number} with best_mse={best:.6f}")
            return float(best)

    return objective


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML config (HPTYaml).",
    ),
    device: str = typer.Option("cuda", help="Torch device string (e.g. cuda, cpu)."),
    num_epochs: int = typer.Option(3, min=1, help="Epochs per trial."),
    log_every_steps: int = typer.Option(10, min=1, help="Log every N steps."),
):
    hpt = load_hpt_config(config)

    datamodule = MNISTDataModule(
        batch_size=hpt.dl_cfg.batch_size,
        data_path=Path(hpt.dl_cfg.data_path),
        num_workers=hpt.dl_cfg.num_workers,
        transform=hpt.dl_cfg.transform,
        shuffle=hpt.dl_cfg.shuffle,
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    study = create_study_from_cfg(hpt)

    objective = make_objective(
        train_loader=train_loader,
        val_loader=val_loader,
        hpt=hpt,
        experiment_name=hpt.opt_study_cfg.study_name or "optuna_hpt",
        device=device,
        num_epochs=num_epochs,
        log_every_steps=log_every_steps,
    )

    study.optimize(objective, n_trials=hpt.opt_study_cfg.n_trials)


if __name__ == "__main__":
    app()
