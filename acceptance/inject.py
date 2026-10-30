"""inject.py SOAK T0 KILL_AT_MIN CRASH_AT_MIN: the soak's fault injector (run as root by soak-remote.sh).

  1. at T0 + KILL_AT_MIN: SIGKILL drumsd's main process. systemd restarts it (Restart=always); drumsd reconciles its envs
     from the journal. Recorded: old/new pid, seconds until /healthz answers again.
  2. at T0 + CRASH_AT_MIN: wait for a slot that drumsd is restoring (state "creating" or "resetting": a new env, a fork
     child or a reset) and kill every process in its cgroups (cgroup.kill, every 20 ms) until drumsd leaves that state:
     the crashed-restore mechanism of drumsd's own recovery_host.py test C. Recorded: slot, state before/after, kills.
Every event is a line in SOAK/inject.jsonl. Nothing else is touched."""
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

SOAK, T0, KILL_AT, CRASH_AT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
URL = "http://127.0.0.1:7070"
TOKEN = open("/etc/drums/token").read().strip()
STATE_DIR = json.load(open("/etc/drums/drumsd.json")).get("state_dir", "/var/lib/drums/state") if os.path.exists("/etc/drums/drumsd.json") else "/var/lib/drums/state"
MARK = hashlib.sha256(STATE_DIR.encode()).hexdigest()[:8]


def log(**ev):
    ev["t"] = time.time(); ev["t_rel_min"] = round((ev["t"] - T0) / 60, 2)
    with open(os.path.join(SOAK, "inject.jsonl"), "a") as f:
        f.write(json.dumps(ev, default=str) + "\n")
    print(json.dumps(ev, default=str), flush=True)


def api(path):
    req = urllib.request.Request(URL + path, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main_pid():
    return int(subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", "drumsd"], capture_output=True, text=True).stdout.strip() or 0)


def healthy():
    try:
        with urllib.request.urlopen(URL + "/healthz", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_until(t_abs):
    while time.time() < t_abs:
        time.sleep(min(5, max(0.05, t_abs - time.time())))


def slot_number(rec):
    """The slot number names the slot's cgroups (drumsbench-rootfs-<mark>-<n>-pool.scope): the record's `num`. (Ids are not
    always "s<num>": fork children are "s100-2".)"""
    if isinstance(rec.get("num"), int):
        return rec["num"]
    sid = str(rec.get("id") or "")
    return int(sid[1:]) if sid[:1] == "s" and sid[1:].isdigit() else None


def slots():
    s = api("/v1/slots")
    return s.get("slots", s) if isinstance(s, dict) else s


def kill_drumsd():
    old = main_pid(); t = time.time()
    os.kill(old, signal.SIGKILL)
    new = old
    while time.time() - t < 120 and (new in (0, old) or not healthy()):
        time.sleep(0.1); new = main_pid()
    log(event="drumsd_sigkill", old_pid=old, new_pid=new, healthy_after_s=round(time.time() - t, 2), healthy=healthy())


def crash_restore(deadline_s=1800):
    t = time.time()
    while time.time() - t < deadline_s:
        try:
            cands = [r for r in slots() if str(r.get("state", "")).lower() in ("creating", "resetting")]
        except Exception as e:  # noqa: BLE001
            cands = []; err = str(e)
        for r in cands:
            n = slot_number(r)
            if n is None:
                log(event="crash_restore_no_slot_number", record_keys=sorted(r)); return
            pool = glob.glob(f"/sys/fs/cgroup/system.slice/*/drumsbench-rootfs-{MARK}-{n}-pool.scope")
            kills, t1, state = 0, time.time(), r.get("state")
            while time.time() - t1 < 120:
                for d in glob.glob(f"/sys/fs/cgroup/system.slice/*/drumsbench-rootfs-{MARK}-{n}-pool.scope/*.scope"):
                    try:
                        if open(d + "/cgroup.procs").read().strip():
                            open(d + "/cgroup.kill", "w").write("1"); kills += 1
                    except OSError:
                        pass
                try:
                    cur = api(f"/v1/slots/{r.get('id')}")
                    state = cur.get("state")
                except Exception:  # noqa: BLE001
                    pass
                if str(state).lower() not in ("creating", "resetting"):
                    break
                time.sleep(0.02)
            log(event="crashed_restore", slot=r.get("id"), slot_number=n, state_before=r.get("state"), state_after=state,
                kills=kills, seconds=round(time.time() - t1, 2), pool_cgroup=pool)
            return
        time.sleep(0.02)
    log(event="crash_restore_no_restore_seen", waited_s=deadline_s)


log(event="injector_start", kill_at_min=KILL_AT, crash_at_min=CRASH_AT, marker=MARK, drumsd_pid=main_pid())
for at, fn in sorted([(KILL_AT, kill_drumsd), (CRASH_AT, crash_restore)], key=lambda x: x[0]):
    if at < 0:
        continue
    wait_until(T0 + at * 60)
    fn()
log(event="injector_done")
