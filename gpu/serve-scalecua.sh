#!/usr/bin/env bash
# serve-scalecua.sh: start prime-rl's inference server with ScaleCUA on the given GPUs, detached, and wait until it answers
# a tool-calling chat request. Prints the endpoint.
#   gpu/serve-scalecua.sh --model <weights dir> --out <log dir> [--gpus 6,7] [--prefix /opt/prime-rl-gpu] [--port 8000]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX=/opt/prime-rl-gpu; MODEL=""; OUT=""; GPUS=6,7; PORT=8000
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;; --out) OUT=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --prefix) PREFIX=$2; shift 2;;
    --port) PORT=$2; shift 2;; -h|--help) sed -n 2,5p "$0"; exit 0;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -f "$MODEL/config.json" ] && [ -n "$OUT" ] || { echo "--model DIR (with config.json) and --out DIR are required" >&2; exit 2; }
mkdir -p "$OUT"; say() { echo "$(date -u +%FT%TZ) [serve] $*" | tee -a "$OUT/serve.log"; }
NDP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$OUT/nvidia-smi-before.csv"
say "inference: model $MODEL on GPUs $GPUS (data parallel $NDP), port $PORT"
cd "$PREFIX/prime-rl"
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "$PREFIX/prime-rl/.venv/bin/inference" @ "$HERE/serve-scalecua.toml" \
  --vllm.model "$MODEL" --vllm.data-parallel-size "$NDP" --server.port "$PORT" --output-dir "$OUT/inference-out" \
  > "$OUT/inference.log" 2>&1 < /dev/null &
echo $! > "$OUT/inference.pid"
t0=$(date +%s)
until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" 2>/dev/null; do
  kill -0 "$(cat "$OUT/inference.pid")" 2>/dev/null || { say "FAILED: inference exited"; tail -40 "$OUT/inference.log"; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 1800 ] || { say "FAILED: not ready after 1800 s"; exit 1; }
  sleep 5
done
say "ready after $(( $(date +%s) - t0 )) s: $(cat "$OUT/models.json")"
# one tool-calling request: the endpoint must return a parsed tool call (the acceptance harness depends on it)
curl -sf -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"scalecua",
 "messages":[{"role":"user","content":"Click the point (10, 20) on the screen."}],"max_tokens":256,"temperature":0,
 "tools":[{"type":"function","function":{"name":"computer_click","description":"Click at x, y","parameters":{"type":"object",
 "properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]}}}]}' > "$OUT/toolcall-check.json"
python3 -c "import json,sys; r=json.load(open('$OUT/toolcall-check.json')); m=r['choices'][0]['message']; print('tool_calls:', json.dumps(m.get('tool_calls'))[:300]); print('content:', (m.get('content') or '')[:200])" | tee -a "$OUT/serve.log"
IP=$(hostname -I | awk '{print $1}')
say "endpoint http://$IP:$PORT/v1 (model scalecua)"
