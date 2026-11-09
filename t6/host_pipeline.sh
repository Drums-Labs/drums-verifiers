#!/usr/bin/env bash
# host_pipeline.sh RELEASE_TAR SEED WANT_TASKS_SHA256 [WORKERS]: the whole replication on the host itself (survives the
# operator's laptop going away): install the release, build + hash-check the task set, install the verifier, validate
# every task, then the hybrid/MCP live check and the low-concurrency rerun of run 1's infra rejects (ids in ~/t6/infra-ids.txt).
# Writes ~/t6/runs/{repro,hybrid.json,infra-rerun-w2}/ and ~/t6/PIPELINE_DONE.
set -uo pipefail
REL=$1; SEED=$2; WANT=$3; WORKERS=${4:-16}
exec > >(tee -a ~/t6/pipeline.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [pipeline] $*"; }
D=$(basename "$REL" .tar.gz)
say "install $D"; cd ~ && tar xzf "$REL" && cd "$D" && sudo ./install.sh > ~/install.out 2>&1 || { say "install failed"; tail -20 ~/install.out; exit 1; }
tail -2 ~/install.out
say "factory"; bash ~/t6/t6/remote_repro.sh "$SEED" "$WANT" || { say "factory failed"; exit 1; }
cd ~/t6 && mkdir -p runs
say "validation, $WORKERS workers"; venv/bin/python -m drums_tasks.validate --tasks ts --out runs/repro --workers "$WORKERS" > runs/repro.log 2>&1; tail -10 runs/repro.log
say "hybrid + MCP"; venv/bin/python -I t6/hybrid_e2e.py --tasks ts --out runs/hybrid.json --n 3 > runs/hybrid.log 2>&1; tail -4 runs/hybrid.log
if [ -s infra-ids.txt ]; then say "infra rerun, 2 workers"; venv/bin/python -m drums_tasks.validate --tasks ts --out runs/infra-rerun-w2 --workers 2 --ids infra-ids.txt > runs/infra-rerun-w2.log 2>&1; tail -8 runs/infra-rerun-w2.log; fi
sudo cat /var/lib/drums/install-receipt.json > runs/install-receipt.json; uname -a > runs/uname.txt
say "done"; touch ~/t6/PIPELINE_DONE
