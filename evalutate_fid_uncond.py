from functools import partial
import mlflow
import torch
import numpy as np
import tempfile
import os
import json

from utils.mlflow_tracking_utils import get_run_param, parse_int_list
from utils.FID.fid_evaluation import evaluate_fid_with_registered_backbone
from models.ode_solvers import get_ode_solver_from_name, sample_unconditional

def run_eval(
    generator_run_id: str,
    generator_name = "UNet",
    backbone_uri: str = "models:/fid_backbone/latest",
    stats_filename: str = "real_stats.npz",
    real_embeddings_filename: str = "real_embeddings.npz",
    n_samples: int = 5210,
    ode_solver_name: str | None = None,
    image_shape: list[int] | tuple[int, ...] | None = None,
    ode_steps: int | None = None,
    batch_size: int | None = None,
    device = "cuda",
    seed:int | None = None,
    export_path: str | None = None,
    real_loader=None,
    show_progress: bool = True):

    if seed:
        torch.manual_seed(seed)
    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    generator = mlflow.pytorch.load_model(f"runs:/{generator_run_id}/{generator_name}").to(device).eval()
    if ode_solver_name is None:
        ode_solver_name = get_run_param(generator_run_id, "ode_solver")
    ode_solver = get_ode_solver_from_name(ode_solver_name)
    if image_shape is None:
        image_shape = parse_int_list(get_run_param(generator_run_id, "image_shape"))
    else:
        image_shape = list(image_shape)
    if ode_steps is None:
        ode_steps = int(get_run_param(generator_run_id, "ode_steps"))
    if batch_size is None:
        batch_size = int(get_run_param(generator_run_id, "batch_size"))


    sample_fn = partial(
        sample_unconditional,
        model=generator,
        image_shape=image_shape,
        ode_solver=ode_solver,
        n_steps=ode_steps,
        seed=None,
        device=device,
    )

    run_name = f"fid::{generator_run_id[:8]}::{ode_solver.__name__}::{ode_steps}steps"
    mlflow.set_experiment("FM-uncond-eval")
    with mlflow.start_run(run_name = run_name):

        fid, gen_embs, real_embs = evaluate_fid_with_registered_backbone(
            sample_fn = sample_fn,
            device = device, 
            backbone_uri=backbone_uri,
            stats_filename=stats_filename,
            real_embeddings_filename=real_embeddings_filename,
            n_samples = n_samples,
            batch_size=batch_size,
            real_loader=real_loader,
            show_progress=show_progress,
        )
        mlflow.log_metric("fid", float(fid))
        mlflow.log_params({
            "fid_backbone_uri": backbone_uri,
            "fid_n_samples": n_samples,
            "fid_batch_size": batch_size,
            "fid_stats_filename": stats_filename,
            "seed": seed
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_path = os.path.join(tmpdir, "generated_embeddings.npz")
            np.savez(gen_path, embs=gen_embs)
            mlflow.log_artifact(gen_path)

            if real_embs is not None:
                real_path = os.path.join(tmpdir, "real_embeddings.npz")
                np.savez(real_path, embs=real_embs)
                mlflow.log_artifact(real_path)

            fid_path = os.path.join(tmpdir, "fid.json")
            with open(fid_path, "w", encoding="utf-8") as f:
                json.dump({"fid": float(fid)}, f)
            mlflow.log_artifact(fid_path)

        if export_path is not None:
            os.makedirs(export_path, exist_ok=True)
            np.savez(os.path.join(export_path, "generated_embeddings.npz"), embs=gen_embs)
            if real_embs is not None:
                np.savez(os.path.join(export_path, "real_embeddings.npz"), embs=real_embs)
            with open(os.path.join(export_path, "fid_eval_fid.json"), "w", encoding="utf-8") as f:
                json.dump({"fid": float(fid)}, f)

        return float(fid), gen_embs, real_embs
        

if __name__ == "__main__":
    run_eval()
