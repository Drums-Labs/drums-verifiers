#!/usr/bin/env bash
# acceptance.sh: one command from a laptop. Install Drums on a Linux host with the release's installer, install prime-rl
# (pinned) and this adapter, then run model-driven desktop rollouts through prime-rl's eval pipeline. Everything is recorded
# and pulled back.
#
#   acceptance/acceptance.sh --host drums-exp2 --release dist/drums-release-<commit>.tar.gz --model-url http://10.0.0.5:8000/v1 \
#       [--model NAME] [--api-key-var VAR] [--tasks 4] [--fork 4] [--rollouts 2] [--tool-style scalecua] [--max-turns 15] \
#       [--out DIR] [--skip-install] [-- extra prime-rl eval args]
#
# --model-url mock   runs the scripted mock policy on the host (plumbing check, no model).
# --model-url URL    any OpenAI-compatible chat endpoint (vLLM) REACHABLE FROM THE HOST, with tool calling enabled.
# The runs (prime-rl `eval`, its orchestrator eval pipeline: env server, dispatcher, episodes):
#   A single   N tasks x R independent episodes, one env each
#   B fork     N tasks x 1 episode: parent env, then --fork children from the reached state, one agent each (a GRPO group)
#   C control  1 task, fork 2 after starting a process that exits right after the capture: every child must be DROPPED
#              (drumsd alive gate), with no reward
# Pass criteria (summary.json): every trace either valid with a checker reward, or dropped with a drums_invalid reason and no
# reward; run C 100% dropped; every trace has screenshots in its tool results; fork children are distinct envs of one parent.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
HOST=""; REL=""; URL=""; MODEL=""; KEYVAR="ACCEPT_API_KEY"; N=4; FORK=4; R=2; STYLE="scalecua"; TURNS=15; OUT=""; SKIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=$2; shift 2;; --release) REL=$2; shift 2;; --model-url) URL=$2; shift 2;; --model) MODEL=$2; shift 2;;
    --api-key-var) KEYVAR=$2; shift 2;; --tasks) N=$2; shift 2;; --fork) FORK=$2; shift 2;; --rollouts) R=$2; shift 2;;
    --tool-style) STYLE=$2; shift 2;; --max-turns) TURNS=$2; shift 2;; --out) OUT=$2; shift 2;; --skip-install) SKIP=1; shift;;
    --) shift; break;; -h|--help) sed -n 2,22p "$0"; exit 0;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$HOST" ] && [ -n "$URL" ] || { echo "--host and --model-url are required" >&2; exit 2; }
[ "$SKIP" = 1 ] || [ -f "$REL" ] || { echo "--release TAR is required unless --skip-install" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%MZ); OUT=${OUT:-$PWD/acceptance-$TS}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/acceptance.log") 2>&1
say() { echo "$(date -u +%FT%TZ) [acceptance] $*"; }
say "host $HOST, model $URL ${MODEL:+($MODEL)}, tasks $N, rollouts $R, fork $FORK, tool style $STYLE, out $OUT"
RD=/opt/drums-acceptance/$TS
ssh "$HOST" "sudo mkdir -p $RD && sudo chown \$(id -u):\$(id -g) $RD"
# the adapter at this exact commit (a bundle: the private repo needs no credentials on the host)
ADAPTER_C=$(git -C "$PKG" rev-parse HEAD)
[ -z "$(git -C "$PKG" status --porcelain --untracked-files=no)" ] || { echo "drums-verifiers has uncommitted changes" >&2; exit 1; }
git -C "$PKG" bundle create -q "$OUT/drums-verifiers.bundle" HEAD
scp -q "$OUT/drums-verifiers.bundle" "$HERE/remote.sh" "$HERE/summarize.py" "$HOST:$RD/"
if [ "$SKIP" = 0 ]; then
  sha256sum "$REL" 2>/dev/null > "$OUT/release.sha256" || shasum -a 256 "$REL" > "$OUT/release.sha256"
  scp -q "$REL" "$HOST:$RD/"
fi
ssh "$HOST" "bash $RD/remote.sh $RD $SKIP $(basename "${REL:-none}") $ADAPTER_C $(printf %q "$URL") $(printf %q "${MODEL:-}") $KEYVAR $N $FORK $R $STYLE $TURNS $(printf '%q ' "$@")" \
  || say "remote.sh exited non-zero (kept going to pull the evidence)"
say "pulling evidence"
rsync -a "$HOST:$RD/" "$OUT/remote/" --exclude '*.tar.gz' --exclude prime-rl --exclude venv
python3 -c "import json; s=json.load(open('$OUT/remote/summary.json')); print(json.dumps(s['verdict'], indent=1))" || true
say "done: $OUT"
