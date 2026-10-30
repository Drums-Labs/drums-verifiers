#!/usr/bin/env bash
# remote.sh: the host side of acceptance.sh (run by it over ssh; not meant to be run by hand).
#   remote.sh RD SKIP_INSTALL RELEASE_TAR ADAPTER_COMMIT MODEL_URL MODEL KEYVAR N FORK R STYLE TURNS [extra eval args]
set -uo pipefail
RD=$1 SKIP=$2 REL=$3 ADAPTER_C=$4 URL=$5 MODEL=$6 KEYVAR=$7 N=$8 FORK=$9 R=${10} STYLE=${11} TURNS=${12}; shift 12
EXTRA=("$@")
PRIME_RL_COMMIT=a566e5d8e2a903f44d901e114d5d510888b93303     # verifiers 60290d43 is its deps/verifiers submodule
TASKS=(e0df059f-28a6-4169-924f-b9623e7184cc 035f41ba-6653-43ab-aa63-c86d449d62e5 2ad9387a-65d8-4e33-ad5b-7580065a27ca
       a4d98375-215b-4a4d-aee9-3d4370fccc41 bedcedc4-4d72-425e-ad62-21960b11fe0d f9be0997-4b7c-45c5-b05c-4612b44a6118
       ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3 28cc3b7e-b194-4bc9-8353-d04c0f4d56d2 94d95f96-9699-4208-98ba-3c3119edf9c2
       2cd43775-7085-45d8-89fa-9e35c0a915cf 3ce045a0-877b-42aa-8d2c-b4a863336ab8)   # all set up valid on drumsd (2026-10-30)
cd "$RD"
exec > >(tee -a "$RD/remote.log") 2>&1
say() { echo "$(date -u +%FT%TZ) [remote] $*"; }
phase() { echo "{\"phase\":\"$1\",\"seconds\":$(( $(date +%s) - $2 )),\"rc\":$3,\"end_utc\":\"$(date -u +%FT%TZ)\"}" >> "$RD/phases.jsonl"; }
fail() { say "FAILED: $*"; echo "{\"failed\":\"$*\"}" > "$RD/failed.json"; exit 1; }
uname -a > host.txt; cat /etc/os-release >> host.txt; nproc >> host.txt; free -g >> host.txt; df -h / >> host.txt
TOK=$(curl -sf -m 2 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token || true)
[ -n "$TOK" ] && for k in instance-id instance-type ami-id placement/availability-zone; do
  echo "$k: $(curl -sf -m 2 -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/$k)"; done >> host.txt

# 1. Drums, with the release's one-command installer
if [ "$SKIP" = 0 ]; then
  t=$(date +%s); say "install: $REL"
  tar xzf "$REL" && D="${REL%.tar.gz}"
  ( time sudo "./$D/install.sh" ) > install.out 2>&1; rc=$?
  sed -i -E 's/(token: +)[^ ]+/\1[REDACTED]/' install.out
  phase install "$t" "$rc"
  sudo cp /var/lib/drums/install-receipt.json /var/lib/drums/install-phases.jsonl . 2>/dev/null; sudo chown "$(id -u)" install-receipt.json install-phases.jsonl 2>/dev/null
  [ "$rc" = 0 ] || fail "install.sh rc=$rc"
fi
sudo install -m 0600 -o "$(id -u)" -g "$(id -g)" /etc/drums/token "$RD/.drums-token"
trap 'rm -f "$RD/.drums-token"' EXIT
export DRUMS_URL=http://127.0.0.1:7070 DRUMS_TOKEN_FILE="$RD/.drums-token"

# 2. prime-rl (pinned, CPU install: no GPU extras) + the drums SDK + this adapter
t=$(date +%s)
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > uv-install.log 2>&1
UV=$(command -v uv || echo ~/.local/bin/uv)
if [ ! -d prime-rl ]; then
  git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git && git -C prime-rl checkout -q "$PRIME_RL_COMMIT" \
    && git -C prime-rl -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init -q -- deps/verifiers deps/renderers deps/pydantic-config deps/prime-envs
fi
(cd prime-rl && "$UV" sync -q) > uv-sync.log 2>&1 || fail "prime-rl uv sync"
git -C prime-rl rev-parse HEAD > versions.txt; git -C prime-rl submodule status >> versions.txt
cp -r /opt/drums/src/drumsd/python drums-sdk-python && "$UV" pip install -q -p prime-rl/.venv/bin/python ./drums-sdk-python >> uv-sync.log 2>&1 || fail "drums SDK install"
rm -rf drums-verifiers && git clone -q drums-verifiers.bundle drums-verifiers && git -C drums-verifiers checkout -q "$ADAPTER_C" || fail "adapter checkout"
"$UV" pip install -q -p prime-rl/.venv/bin/python --no-deps -e ./drums-verifiers >> uv-sync.log 2>&1 || fail "adapter install"
echo "drums-verifiers $ADAPTER_C" >> versions.txt
prime-rl/.venv/bin/python -c "import drums, verifiers; print('drums SDK', drums.__version__)" >> versions.txt
sudo cat /var/lib/drums/install-receipt.json 2>/dev/null | python3 -c "import json,sys; r=json.load(sys.stdin); print('drumsd', r['drumsd_commit'], 'criu', r['lock']['sources']['drums-criu']['commit'], 'criu sha256', r['criu_sha256'])" >> versions.txt
phase setup "$t" 0

# 3. the model: the given endpoint, or the scripted mock on this host
MOCK_PID=""
if [ "$URL" = mock ]; then
  prime-rl/.venv/bin/python drums-verifiers/examples/mock_policy.py --port 8011 --log "$RD/mock_calls.jsonl" > mock.log 2>&1 &
  MOCK_PID=$!; URL=http://127.0.0.1:8011/v1; MODEL=${MODEL:-mock}; sleep 1
fi
export "$KEYVAR"="${!KEYVAR:-EMPTY}"
curl -sf -m 20 -H "Authorization: Bearer ${!KEYVAR}" "$URL/models" > models.json || fail "model endpoint $URL/models not reachable from this host"
[ -n "$MODEL" ] || MODEL=$(python3 -c "import json; print(json.load(open('models.json'))['data'][0]['id'])")
say "model $MODEL at $URL"
prime-rl/.venv/bin/python - > drumsd-before.json <<'EOF'
import json, drums_verifiers.sdk as s
c = s.client(); print(json.dumps({"health": c.health(), "metrics": c.metrics()}, default=str))
EOF

# 4. the runs, through prime-rl's own eval entrypoint
SEL=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${TASKS[@]:0:$N}")
run() { # run NAME [eval args...]
  local name=$1; shift; local t; t=$(date +%s); say "run $name"
  mkdir -p "runs/$name"
  PATH="$RD/prime-rl/.venv/bin:$PATH" DRUMS_RECORD="$RD/runs/$name/record.jsonl" "$RD/prime-rl/.venv/bin/eval" drums-verifiers \
    -m "$MODEL" --client.base-url "$URL" --client.api-key-var "$KEYVAR" --no-monitors.prime --output-dir "$RD/runs/$name/out" \
    --env.taskset.task.tools.style "$STYLE" --env.agent.max-turns "$TURNS" "$@" "${EXTRA[@]}" > "runs/$name/eval.log" 2>&1
  local rc=$?; phase "run-$name" "$t" "$rc"; say "run $name rc=$rc"
}
run A-single -n "$N" -r "$R" -c "$((N * R))" --env.taskset.task-ids "$SEL"
run B-fork -n "$N" -r 1 -c "$N" --env.taskset.task-ids "$SEL" --env.fork-n "$FORK" --env.settle-s 3
CTRL='[{"type": "launch", "command": ["python3", "-c", "import time\nT = time.time() + 7\nwhile time.time() < T: time.sleep(0.2)"]}]'
run C-control -n 1 -r 1 -c 1 --env.taskset.task-ids "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:2]))" "${TASKS[@]}")" \
  --env.fork-n 2 --env.prefix "$CTRL" --env.settle-s 0
prime-rl/.venv/bin/python - > drumsd-after.json <<'EOF'
import json, drums_verifiers.sdk as s
c = s.client(); print(json.dumps({"health": c.health(), "metrics": c.metrics(), "envs": [e.record for e in c.envs()]}, default=str))
EOF
[ -n "$MOCK_PID" ] && kill "$MOCK_PID"
sudo journalctl -u drumsd --since "-3h" --no-pager > drumsd-journal.txt 2>/dev/null
prime-rl/.venv/bin/python summarize.py "$RD" > summary.txt; cat summary.txt
# the token never leaves the host: scrub it from every text file before acceptance.sh pulls the directory
T=$(cat "$RD/.drums-token"); grep -rlF "$T" "$RD" --exclude=.drums-token 2>/dev/null | while read -r f; do sed -i "s|$T|[REDACTED]|g" "$f"; echo "scrubbed $f"; done
