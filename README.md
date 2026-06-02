```bash
mlflow ui --backend-store-uri sqlite:////home/luke-padmore/Source/flow-matching-mnist/mlflow.db --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI="sqlite:////home/luke-padmore/Source/flow-matching-mnist/mlflow.db"

export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
mlflow ui --backend-store-uri "file:$(pwd)/mlruns" --host 127.0.0.1 --port 5000

```