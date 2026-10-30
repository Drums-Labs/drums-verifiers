#!/usr/bin/env bash
# soak.sh: the acceptance workflow in a loop for a fixed time, unattended, with faults injected mid-run. From a laptop:
#
#   acceptance/soak.sh --host drums-acceptance --base /opt/drums-acceptance/<install-only ts> --model-url http://<ip>:8000/v1 \
#       [--model scalecua] [--minutes 90] [--kill-drumsd-at 30] [--crash-restore-at 60] [--tasks 4] [--fork 4] [--rollouts 2]
#       [--max-turns 15] [--out DIR]
#   acceptance/soak.sh --host drums-acceptance --pull /opt/drums-soak/<ts> [--out DIR]      # fetch the evidence afterwards
#
# The loop runs on the host under setsid (soak-remote.sh): closing the laptop does not stop it. Each iteration is a full
# acceptance run (single episodes, fork groups, negative control), so every iteration has its own pass/fail summary.
# Injected: drumsd SIGKILLed once (systemd restarts it; envs are reconciled from the journal) and one crashed restore (a slot's
# cgroups killed while CRIU restores it). soak_summary.py reports episodes, valid rewards, invalid-as-valid (cross-checked
# against drumsd's own verify events), recoveries, and every command that was not issued by the loop itself (none).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(cd "$HERE/.." && pwd)"
HOST=""; BASE=""; URL=""; MODEL=""; KEYVAR="ACCEPT_API_KEY"; MIN=90; KILL_AT=30; CRASH_AT=60; N=4; FORK=4; R=2; TURNS=15
STYLE=scalecua; KEEP=5; TEMP=0.8; MAXTOK=2048; PAUSE=2; OUT=""; PULL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=$2; shift 2;; --base) BASE=$2; shift 2;; --model-url) URL=$2; shift 2;; --model) MODEL=$2; shift 2;;
    --api-key-var) KEYVAR=$2; shift 2;; --minutes) MIN=$2; shift 2;; --kill-drumsd-at) KILL_AT=$2; shift 2;;
    --crash-restore-at) CRASH_AT=$2; shift 2;; --tasks) N=$2; shift 2;; --fork) FORK=$2; shift 2;; --rollouts) R=$2; shift 2;;
    --max-turns) TURNS=$2; shift 2;; --out) OUT=$2; shift 2;; --pull) PULL=$2; shift 2;;
    -h|--help) sed -n 2,16p "$0"; exit 0;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$HOST" ] || { echo "--host is required" >&2; exit 2; }
if [ -n "$PULL" ]; then
  OUT=${OUT:-$PWD/soak-$(basename "$PULL")}; mkdir -p "$OUT"
  rsync -a "$HOST:$PULL/" "$OUT/" --exclude prime-rl --exclude '*.bundle'
  cat "$OUT/soak-summary.txt" 2>/dev/null || echo "no soak-summary.txt yet (still running?)"; exit 0
fi
[ -n "$BASE" ] && [ -n "$URL" ] || { echo "--base and --model-url are required" >&2; exit 2; }
[ -z "$(git -C "$PKG" status --porcelain --untracked-files=no)" ] || { echo "drums-verifiers has uncommitted changes" >&2; exit 1; }
ADAPTER_C=$(git -C "$PKG" rev-parse HEAD)
TS=$(date -u +%Y%m%dT%H%MZ); SOAK=/opt/drums-soak/$TS
TMP=$(mktemp -d); git -C "$PKG" bundle create -q "$TMP/drums-verifiers.bundle" HEAD
ssh "$HOST" "sudo mkdir -p $SOAK && sudo chown \$(id -u):\$(id -g) $SOAK && test -x $BASE/prime-rl/.venv/bin/eval"
scp -q "$TMP/drums-verifiers.bundle" "$HERE/remote.sh" "$HERE/summarize.py" "$HERE/soak-remote.sh" "$HERE/inject.py" "$HERE/soak_summary.py" "$HOST:$SOAK/"
rm -rf "$TMP"
ssh "$HOST" "cd $SOAK || exit 1; setsid nohup bash soak-remote.sh $SOAK $BASE $MIN $KILL_AT $CRASH_AT $ADAPTER_C $(printf %q "$URL") $(printf %q "${MODEL:-}") $KEYVAR $N $FORK $R $STYLE $TURNS $KEEP $TEMP $MAXTOK $PAUSE > soak.stdout 2>&1 < /dev/null & echo started"
echo "soak running on $HOST in $SOAK (adapter $ADAPTER_C); pull with: $0 --host $HOST --pull $SOAK"
