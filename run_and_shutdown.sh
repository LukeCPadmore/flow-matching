#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/train/mnist_uncond.yaml"
DO_SHUTDOWN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG_PATH="$2"; shift 2 ;;
    --shutdown) DO_SHUTDOWN=true; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: ./run_and_shutdown.sh [--config PATH] [--shutdown]
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SESSION="train_$(basename "${CONFIG_PATH%.*}")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists"
  exit 1
fi

echo "Starting MLflow Docker Compose stack"
(cd "$HOME/ml-storage" && docker compose up -d)

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail

echo 'Initialising conda'
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate ml

echo 'Python:'
which python
python -c 'import sys; print(sys.executable)'

echo \"Starting job at \$(date)\"

set +e
python train.py fit --config \"$CONFIG_PATH\"
EXIT_CODE=\$?
set -e

echo \"Job finished at \$(date) with exit code \$EXIT_CODE\"

if $DO_SHUTDOWN; then
  echo 'Shutting down...'
  sudo /sbin/shutdown -h now
fi

exit \$EXIT_CODE
"

echo "Started tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
echo "Shutdown after completion: $DO_SHUTDOWN"
