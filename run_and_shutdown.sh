#!/usr/bin/env bash
set -euo pipefail

SESSION="train_uncond"
CMD="python train.py fit --config configs/train/mnist_uncond.yaml"

DO_SHUTDOWN=false
KEEP_SESSION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shutdown) DO_SHUTDOWN=true; shift ;;
    --keep)     KEEP_SESSION=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists"
  exit 1
fi

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
$CMD
EXIT_CODE=\$?
set -e

echo \"Job finished at \$(date) with exit code \$EXIT_CODE\"

if $DO_SHUTDOWN; then
  echo 'Shutting down...'
  sudo /sbin/shutdown -h now
elif ! $KEEP_SESSION; then
  echo 'Killing tmux session...'
  tmux kill-session -t \"$SESSION\"
else
  echo 'Keeping tmux session alive.'
fi

exit \$EXIT_CODE
"

echo "Started tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
echo "Shutdown after completion: $DO_SHUTDOWN"
echo "Keep session after completion: $KEEP_SESSION"
