from pathlib import Path
import ast
from mlflow.tracking import MlflowClient

def get_run_param(run_id: str, key: str) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    if key not in run.data.params:
        raise KeyError(f"Run {run_id} has no param '{key}'. Available: {list(run.data.params)[:20]} ...")
    return run.data.params[key] 

def parse_int_list(s: str) -> tuple[int]:
    obj = ast.literal_eval(s)
    if isinstance(obj, tuple):
        obj = tuple(obj)
    if not isinstance(obj, tuple):
        raise ValueError("Expected list or tuple")
    if not all(isinstance(x, int) for x in obj):
        raise ValueError("Non-integer element found")
    return obj


def load_lightning_checkpoint_path_from_run(
    run_id: str,
    artifact_path: str = "checkpoints/best.ckpt",
    *,
    dst_path: str | None = None,
) -> Path:
    client = MlflowClient()
    return Path(client.download_artifacts(run_id, artifact_path, dst_path=dst_path))
    
