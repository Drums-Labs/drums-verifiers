#!/usr/bin/env bash
# soak-remote.sh: the host side of soak.sh (started detached by it; no ssh session needed while it runs).
#   soak-remote.sh SOAK_DIR BASE_RD MINUTES KILL_AT_MIN CRASH_AT_MIN ADAPTER_COMMIT MODEL_URL MODEL KEYVAR N FORK R STYLE TURNS KEEP TEMP MAXTOK PAUSE [HARD_STOP_EPOCH]
# Loops the full acceptance workflow (remote.sh: A single episodes, B fork groups, C negative control; every episode
# verified, every env reset/closed by the adapter) until MINUTES have passed, each iteration in SOAK_DIR/iter-NNN with its
# own summary.json. The installed prime-rl venv of BASE_RD (an --install-only run) is reused through a symlink.
# inject.py runs alongside: SIGKILL of drumsd at KILL_AT_MIN, and one crashed restore (a slot's cgroups killed while it is
# being restored) at CRASH_AT_MIN. Nothing here waits for a person; a failed iteration is recorded and the loop goes on.
set -uo pipefail
SOAK=$1 BASE=$2 MIN=$3 KILL_AT=$4 CRASH_AT=$5 ADAPTER_C=$6 URL=$7 MODEL=$8 KEYVAR=$9 N=${10} FORK=${11} R=${12} STYLE=${13} TURNS=${14} KEEP=${15} TEMP=${16} MAXTOK=${17} PAUSE=${18} HARD=${19:-0}
cd "$SOAK"
exec > >(tee -a "$SOAK/soak.log") 2>&1
say() { echo "$(date -u +%FT%TZ) [soak] $*"; }
T0=$(date +%s); echo "$T0" > t0
END=$(( T0 + MIN * 60 ))
[ "$HARD" != 0 ] && [ "$HARD" -lt "$END" ] && END=$HARD
say "hard stop $([ "$HARD" != 0 ] && date -u -d @$HARD +%FT%TZ || echo none)"; say "soak for $MIN min (until $(date -u -d @$END +%FT%TZ)); drumsd kill at +$KILL_AT min, crashed restore at +$CRASH_AT min; model $URL"
sudo -n true || { say "FAILED: needs passwordless sudo"; exit 1; }
sudo systemctl show drumsd -p MainPID -p NRestarts -p Restart > drumsd-before.txt
sudo python3 "$SOAK/inject.py" "$SOAK" "$T0" "$KILL_AT" "$CRASH_AT" > inject.log 2>&1 &
INJ=$!
i=0
while [ "$(date +%s)" -lt "$(( END - 60 ))" ]; do
  i=$((i + 1)); RD=$SOAK/iter-$(printf %03d $i); mkdir -p "$RD"
  ln -sfn "$BASE/prime-rl" "$RD/prime-rl"
  cp "$SOAK/drums-verifiers.bundle" "$SOAK/remote.sh" "$SOAK/summarize.py" "$RD/"
  t=$(date +%s); say "iteration $i start ($RD)"
  bash "$RD/remote.sh" "$RD" 1 none "$ADAPTER_C" "$URL" "$MODEL" "$KEYVAR" "$N" "$FORK" "$R" "$STYLE" "$TURNS" 0 "$HARD" "$KEEP" "$TEMP" "$MAXTOK" "$PAUSE" > "$RD/remote.stdout" 2>&1
  rc=$?
  pass=$(python3 -c "import json; print(json.load(open('$RD/summary.json'))['verdict']['pass'])" 2>/dev/null || echo none)
  echo "{\"iter\": $i, \"start\": $t, \"end\": $(date +%s), \"rc\": $rc, \"pass\": \"$pass\"}" >> "$SOAK/iterations.jsonl"
  say "iteration $i rc=$rc pass=$pass ($(( $(date +%s) - t )) s)"
done
kill "$INJ" 2>/dev/null; wait "$INJ" 2>/dev/null
sudo systemctl show drumsd -p MainPID -p NRestarts -p Restart > drumsd-after.txt
sudo journalctl -u drumsd --since "@$T0" --no-pager > drumsd-journal-soak.txt 2>/dev/null
sudo cp -a /var/lib/drums/journal/events.jsonl drumsd-events.jsonl 2>/dev/null; sudo chown "$(id -u)" drumsd-events.jsonl 2>/dev/null
python3 "$SOAK/soak_summary.py" "$SOAK" > soak-summary.txt 2>&1; cat soak-summary.txt
T=$(sudo cat /etc/drums/token); grep -rlF "$T" "$SOAK" 2>/dev/null | while read -r f; do sed -i "s|$T|[REDACTED]|g" "$f"; done
say "soak done"; touch "$SOAK/DONE"
