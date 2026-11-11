#!/usr/bin/env bash
# reproduce.sh HOST OUT_DIR: rebuild the T6 headline (task set + checker validation) on a fresh Ubuntu 24.04 host.
#
#   HOST     an ssh alias for a fresh c8i.16xlarge (or larger) Ubuntu 24.04 x86_64 host, kernel 7.0.0-1014-aws, 200 GB disk
#   OUT_DIR  where the evidence lands on this machine
# Env (pins; defaults are the T6 run's):
#   DRUMSD_COMMIT VERIFIERS_COMMIT SEED (20261108) WORKERS (16)
# Needs: git and gh access to Drums-Labs (drumsd, drums-criu, drums-lab, drums-verifiers), python3, ssh/scp/rsync.
# Steps: build the release from DRUMSD_COMMIT (make-release.sh), install it on HOST (install.sh), copy drums-verifiers at
# VERIFIERS_COMMIT, build the task set with SEED and check its sha256 against the frozen ones, install the verifier,
# run checker validation on all tasks, pull records + summary. About 15 min install + 40 min validation.
set -euo pipefail
H=${1:?host}; OUT=${2:?out dir}
DRUMSD_COMMIT=${DRUMSD_COMMIT:?set DRUMSD_COMMIT (see raw/MANIFEST.json of the T6 report)}
VERIFIERS_COMMIT=${VERIFIERS_COMMIT:?set VERIFIERS_COMMIT}
SEED=${SEED:-20261108}; WORKERS=${WORKERS:-16}
WANT_TASKS=${WANT_TASKS:-0a41aaa2a641b57472e5d5e969fe2e64e6a0cd737c3ad59dc354267c27fd2e1c}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
say() { echo "$(date -u +%FT%TZ) [reproduce] $*" | tee -a "$OUT/reproduce.log"; }
say "sources"
git clone -q https://github.com/Drums-Labs/drumsd.git "$W/drumsd" && git -C "$W/drumsd" checkout -q "$DRUMSD_COMMIT"
git clone -q --bare https://github.com/Drums-Labs/drums-criu.git "$W/criu.git"
git clone -q --bare https://github.com/Drums-Labs/drums-lab.git "$W/lab.git"
git clone -q https://github.com/Drums-Labs/drums-verifiers.git "$W/dv" && git -C "$W/dv" checkout -q "$VERIFIERS_COMMIT"
say "release"
DRUMS_CRIU="$W/criu.git" DRUMS_LAB="$W/lab.git" bash "$W/drumsd/install/make-release.sh" "$W/rel" | tee -a "$OUT/reproduce.log"
REL=$(ls "$W"/rel/drums-release-*.tar.gz)
scp -q "$REL" "$H:~/"
say "install on $H"
D=$(basename "$REL" .tar.gz)
ssh "$H" "tar xzf $D.tar.gz && cd $D && sudo ./install.sh > ~/install.out 2>&1; tail -3 ~/install.out" | tee -a "$OUT/reproduce.log"
say "factory"
git -C "$W/dv" archive --format=tar HEAD src/drums_tasks src/drums_verifiers/scalecua.py t6 | ssh "$H" "rm -rf ~/t6 && mkdir -p ~/t6 && tar -x -C ~/t6 && mv ~/t6/src/* ~/t6/ && rmdir ~/t6/src"
ssh "$H" "bash ~/t6/t6/remote_repro.sh $SEED $WANT_TASKS" | tee -a "$OUT/reproduce.log"
say "validation ($WORKERS workers)"
ssh "$H" "cd ~/t6 && mkdir -p runs && venv/bin/python -m drums_tasks.validate --tasks ts --out runs/repro --workers $WORKERS > runs/repro.log 2>&1; tail -12 runs/repro.log"
rsync -a "$H:~/t6/runs/repro/" "$OUT/validate/"; rsync -a "$H:~/t6/runs/repro.log" "$H:~/t6/ts/index.json" "$H:~/install.out" "$OUT/"
python3 -c "import json;s=json.load(open('$OUT/validate/summary.json'));print('acceptance', s['accepted'], '/', s['tasks'], s['acceptance_rate'])" | tee -a "$OUT/reproduce.log"
say "done: compare with run 1: python3 t6/compare_runs.py RUN1/records.jsonl $OUT/validate/records.jsonl"
