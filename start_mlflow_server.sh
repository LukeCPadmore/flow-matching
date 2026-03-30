#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env vars or flags)
PROJECT_ROOT="${PROJECT_ROOT:-/home/luke-padmore/Source/flow-matching-mnist}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
DB_PATH="${DB_PATH:-$PROJECT_ROOT/mlflow.db}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/mlruns}"
KILL_PORT=false

usage() {
  cat <<'EOF'
Usage: ./start_mlflow_server.sh [options]

Options:
  --kill-port        Kill any process currently using --port before starting.
  --host <host>      Host to bind (default: 127.0.0.1)
  --port <port>      Port to bind (default: 5000)
  --db-path <path>   SQLite DB file path (default: <PROJECT_ROOT>/mlflow.db)
  --artifact-root <path>
                     Artifact root directory (default: <PROJECT_ROOT>/mlruns)
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill-port)
      KILL_PORT=true
      shift
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$DB_PATH")" "$ARTIFACT_ROOT"

if [[ "$KILL_PORT" == true ]]; then
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" || true
  else
    # Fallback if fuser is unavailable
    if command -v lsof >/dev/null 2>&1; then
      PIDS="$(lsof -t -i:"$PORT" || true)"
      if [[ -n "$PIDS" ]]; then
        kill $PIDS || true
      fi
    fi
  fi
fi

BACKEND_URI="sqlite:///$DB_PATH"
ARTIFACT_URI="file://$ARTIFACT_ROOT"

echo "Starting MLflow server"
echo "  Host:          $HOST"
echo "  Port:          $PORT"
echo "  Backend URI:   $BACKEND_URI"
echo "  Artifact URI:  $ARTIFACT_URI"
echo "  URL:           http://$HOST:$PORT"

exec mlflow server \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$ARTIFACT_URI" \
  --host "$HOST" \
  --port "$PORT"
