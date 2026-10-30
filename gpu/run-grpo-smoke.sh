#!/usr/bin/env bash
# run-grpo-smoke.sh: prime-rl's own `rl` entrypoint (inference + orchestrator + trainer, GRPO) on Drums desktops through
# this adapter, with examples/rl-scalecua-p4d-smoke.toml. Unmodified prime-rl a566e5d8; only config and the environment.
#   gpu/run-grpo-smoke.sh --model <start weights> --out <EBS dir> --drums-url http://<env host>:7070 --drums-token <file>
#       [--gpus 0,1,2,3,4,5] [--prefix /opt/prime-rl-gpu] [--config FILE] [--dry-run] [-- extra rl args]
# Writes OUT/run.log, OUT/record.jsonl (one line per scored trace, from the adapter), OUT/nvidia-smi.csv (5 s samples),
# prime-rl's run dir OUT/<run name>/ (logs, metrics, traces, checkpoints), then gpu/summarize_grpo.py -> OUT/steps.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
PREFIX=/opt/prime-rl-gpu; MODEL=""; OUT=""; DURL=""; DTOK=""; GPUS=0,1,2,3,4,5; CFG="$PKG/examples/rl-scalecua-p4d-smoke.toml"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;; --out) OUT=$2; shift 2;; --drums-url) DURL=$2; shift 2;; --drums-token) DTOK=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;; --prefix) PREFIX=$2; shift 2;; --config) CFG=$2; shift 2;; --dry-run) DRY=1; shift;;
    --) shift; break;; -h|--help) sed -n 2,7p "$0"; exit 0;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -f "$MODEL/config.json" ] && [ -n "$OUT" ] && [ -n "$DURL" ] && [ -f "$DTOK" ] || { echo "--model, --out, --drums-url, --drums-token are required" >&2; exit 2; }
mkdir -p "$OUT"; exec > >(tee -a "$OUT/run.log") 2>&1
say() { echo "$(date -u +%FT%TZ) [grpo-smoke] $*"; }
PY=$PREFIX/prime-rl/.venv/bin/python
say "prime-rl $(git -C "$PREFIX/prime-rl" rev-parse HEAD), adapter $(git -C "$PREFIX/drums-verifiers" rev-parse HEAD 2>/dev/null || echo '?'), config $CFG ($(sha256sum "$CFG" | cut -c1-16))"
cp "$CFG" "$OUT/config.toml"
export DRUMS_URL=$DURL DRUMS_TOKEN_FILE=$DTOK DRUMS_RECORD=$OUT/record.jsonl
"$PY" -c "import drums_verifiers.sdk as s, json; c = s.client(); h = c.health(); print('drumsd', json.dumps(h)[:300])" || { say "FAILED: drumsd not reachable at $DURL"; exit 1; }
for p in 8200 8300; do ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null || { say "FAILED: port $p already in use"; exit 1; }; done
ARGS=(@ "$CFG" --model.name "$MODEL" --output-dir "$OUT" --trainer.model.conversion-dir "$OUT/conversion" --no-monitors.prime "$@")
cd "$PREFIX/prime-rl"
CUDA_VISIBLE_DEVICES=$GPUS "$PREFIX/prime-rl/.venv/bin/rl" "${ARGS[@]}" --dry-run > "$OUT/dry-run.log" 2>&1 || { tail -30 "$OUT/dry-run.log"; say "FAILED: dry run"; exit 1; }
RES=$(ls -d "$OUT"/*/configs/attempt_*/resolved | tail -1)
"$PY" - "$RES" <<'PY'
import json, sys
r = sys.argv[1]
o, i = json.load(open(f"{r}/orchestrator.json")), json.load(open(f"{r}/inference.json"))
port = int(o["model"]["client"]["base_url"].rsplit(":", 1)[1].split("/")[0])
assert port == i["server"]["port"], f"orchestrator samples from port {port}, but the policy server listens on {i['server']['port']}"
print("policy client ->", o["model"]["client"]["base_url"], "admin", o["model"]["client"]["admin_base_url"])
PY
[ "$DRY" = 1 ] && { say "dry run ok: $RES"; exit 0; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$OUT/nvidia-smi.csv" 2>&1 &
SMI=$!
t0=$(date +%s); say "rl start on GPUs $GPUS"
set +e
CUDA_VISIBLE_DEVICES=$GPUS "$PREFIX/prime-rl/.venv/bin/rl" "${ARGS[@]}" > "$OUT/rl.stdout" 2>&1
rc=$?
set -e
kill $SMI 2>/dev/null || true
say "rl exited rc=$rc after $(( $(date +%s) - t0 )) s"
"$PY" "$HERE/summarize_grpo.py" "$OUT" > "$OUT/steps.txt" 2>&1 || true
cat "$OUT/steps.txt"
exit $rc
