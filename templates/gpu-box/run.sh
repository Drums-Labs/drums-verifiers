#!/usr/bin/env bash
# run.sh: prime-rl + Drums desktops on any 8-GPU Linux machine you control, with the same config and runner as the Modal
# template (templates/rl-config.toml, templates/common/drums_rl.py) inside the same image (templates/common/Dockerfile).
#
#   templates/gpu-box/run.sh build                 build the image (docker; about 30-60 min, no GPU needed)
#   templates/gpu-box/run.sh prepare               copy [tasks] train / heldout into $DRUMS_RL_HOME/runs/tasks
#   templates/gpu-box/run.sh download              the pinned weights into $DRUMS_RL_HOME/weights
#   templates/gpu-box/run.sh check                 imports, Drums health, prime-rl dry run
#   templates/gpu-box/run.sh scan                  the untrained model on the training tasks (uses GPUs 0-1)
#   templates/gpu-box/run.sh smoke                 one optimizer step end to end
#   templates/gpu-box/run.sh train [-- args]       the run; restarted up to $DRUMS_RL_RETRIES times on a crash (it resumes)
#   templates/gpu-box/run.sh evaluate --model base --split heldout --k 6 --tag step0-heldout
#   templates/gpu-box/run.sh export [--step N]
#   templates/gpu-box/run.sh spend
#
# Needs: docker with the NVIDIA container toolkit (`docker run --gpus all ... nvidia-smi` works), python3 >= 3.11 on the
# host (to read the config), DRUMS_URL and DRUMS_TOKEN in the environment (never written anywhere by this script), and
# about 1 TB of disk under $DRUMS_RL_HOME for a 27B run (weights 56 GB, two checkpoints of about 333 GB, exports 56 GB).
# Guide: docs/train-on-your-gpus.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TEMPLATES=$(dirname "$HERE")
CONFIG=${DRUMS_RL_CONFIG:-$TEMPLATES/rl-config.toml}
HOME_DIR=${DRUMS_RL_HOME:-/data/drums-rl}
IMAGE=${DRUMS_RL_IMAGE:-drums-rl:local}
RETRIES=${DRUMS_RL_RETRIES:-2}
cmd=${1:-}; [ -n "$cmd" ] || { sed -n 2,20p "$0"; exit 2; }; shift

py() { python3 -c "import sys, tomllib; c = tomllib.load(open(sys.argv[1], 'rb')); $1" "$CONFIG"; }
python3 -c 'import tomllib' 2>/dev/null || { echo "python3 >= 3.11 is needed on the host to read $CONFIG" >&2; exit 2; }

run_in_image() {
  mkdir -p "$HOME_DIR/weights" "$HOME_DIR/runs"
  local env_args=(-e DRUMS_RL_PLATFORM=gpu-box -e DRUMS_RL_CONFIG=/opt/drums-rl/rl-config.toml
                  -e DRUMS_RL_WEIGHTS=/weights -e DRUMS_RL_RUNS=/runs)
  # the names only: docker reads the values from this shell's environment, so they never appear in a command line
  [ -n "${DRUMS_URL:-}" ] && env_args+=(-e DRUMS_URL)
  [ -n "${DRUMS_TOKEN:-}" ] && env_args+=(-e DRUMS_TOKEN)
  [ -n "${HF_TOKEN:-}" ] && env_args+=(-e HF_TOKEN)
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOME_DIR/weights:/weights" -v "$HOME_DIR/runs:/runs" \
    -v "$TEMPLATES/common:/opt/drums-rl/common:ro" -v "$CONFIG:/opt/drums-rl/rl-config.toml:ro" \
    "${env_args[@]}" "$IMAGE" python3 /opt/drums-rl/common/drums_rl.py "$@"
}

case "$cmd" in
  build)
    docker build -t "$IMAGE" -f "$TEMPLATES/common/Dockerfile" \
      --build-arg PRIME_RL_COMMIT="$(py 'print(c["image"]["prime_rl_commit"])')" \
      --build-arg DRUMS_VERIFIERS="$(py 'print(c["image"]["drums_verifiers"])')" \
      --build-arg DRUMS_SDK="$(py 'print(c["image"]["drums_sdk"])')" \
      "$TEMPLATES/common" ;;
  prepare)
    for split in train heldout; do
      src=$(py "import os; p = c['tasks']['$split']; print(p if os.path.isabs(p) else os.path.join(os.path.dirname(sys.argv[1]), p))")
      [ -f "$src/tasks.jsonl" ] || { echo "[tasks] $split = $src: no tasks.jsonl there" >&2; exit 1; }
      mkdir -p "$HOME_DIR/runs/tasks/$split"
      cp "$src/tasks.jsonl" "$HOME_DIR/runs/tasks/$split/tasks.jsonl"
      echo "copied $src/tasks.jsonl -> $HOME_DIR/runs/tasks/$split/"
    done ;;
  train)
    # a crash (not a budget stop, which exits 0) is retried: drums_rl moves partial checkpoints aside and resumes
    n=0
    until run_in_image train "$@"; do
      n=$((n + 1)); [ "$n" -le "$RETRIES" ] || { echo "train failed $n times; see $HOME_DIR/runs/*/logs" >&2; exit 1; }
      echo "train exited non-zero; retry $n of $RETRIES in 60 s (resumes from the last complete checkpoint)"; sleep 60
    done ;;
  download|check|scan|smoke|evaluate|export|spend)
    run_in_image "$cmd" "$@" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
