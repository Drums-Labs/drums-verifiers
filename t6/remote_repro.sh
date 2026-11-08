#!/usr/bin/env bash
# remote_repro.sh SEED WANT_TASKS_SHA256: on the installed host, from ~/t6: factory venv, build, hash check, verifier.
set -euo pipefail
SEED=$1; WANT=$2
sudo cat /etc/drums/token > ~/.drums-token; chmod 600 ~/.drums-token
sudo apt-get install -y -q python3.12-venv > /dev/null
cd ~/t6
python3 -m venv venv
venv/bin/pip -q install -r t6/requirements-factory.txt /opt/drums/sdk/drums-*.whl
venv/bin/python -m drums_tasks.build --seed "$SEED" --out ts > build.json
cat build.json
GOT=$(python3 -c 'import json; print(json.load(open("ts/index.json"))["sha256"]["tasks.jsonl"])')
[ "$GOT" = "$WANT" ] || { echo "tasks.jsonl sha256 $GOT differs from the frozen $WANT" >&2; exit 1; }
sudo install -m 0644 -o root -g root ts/drums_t6.py /etc/drums/verifiers/drums_t6.py
echo "factory ready: tasks.jsonl $GOT"
