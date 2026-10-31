"""soak_summary.py SOAK: totals over every soak iteration (writes SOAK/soak-summary.json, prints it).

  episodes          every trace the adapter recorded (iter-*/runs/*/record.jsonl), by outcome
  valid rewards     traces handed a reward
  invalid_as_valid  rewarded traces that drumsd itself did not verify as valid with that score: each rewarded row
                    (env, drums episode) is looked up in drumsd's own event log (drumsd-events.jsonl, "verify" lines); a row
                    with no matching valid verify event, or a different score, counts. MUST be 0.
  recoveries        per injected fault (inject.jsonl): what drumsd did, the first valid rewarded or verified episode after
                    it, and whether every later iteration still completed
  manual            commands not issued by the loop: none are possible in this design (the loop runs detached); listed as
                    the count of entries in SOAK/manual.log, which a person must append to if they intervene
Per iteration: rc, pass verdict, and the counts."""
import collections
import glob
import json
import os
import sys

SOAK = sys.argv[1]
rows = []
iters = []
for it in sorted(glob.glob(os.path.join(SOAK, "iter-*"))):
    name = os.path.basename(it)
    s = {}
    if os.path.exists(os.path.join(it, "summary.json")):
        s = json.load(open(os.path.join(it, "summary.json")))
    n = collections.Counter()
    for f in sorted(glob.glob(os.path.join(it, "runs", "*", "record.jsonl"))):
        run = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line); r["_iter"] = name; r["_run"] = run
            if r.get("cleanup_failed"):
                n["cleanup_failed"] += 1
                continue
            rows.append(r)
            n["traces"] += 1
            n["valid_rewarded"] += bool(r.get("valid") and r.get("reward") is not None)
            n["dropped"] += bool(r.get("dropped")); n["errored"] += r.get("ok") is False
            n["fork_child_failed"] += bool(r.get("fork_child_failed"))
            n["episodes_failed"] += bool(r.get("episode_failed") or r.get("env_create_failed"))
            n["admission_waits"] += len(r.get("admission_waits") or [])
    iters.append({"iter": name, "pass": (s.get("verdict") or {}).get("pass"),
                  "partial_at_deadline": (s.get("verdict") or {}).get("partial_at_deadline") or [],
                  "failed_checks": [k for k, v in ((s.get("verdict") or {}).get("checks") or {}).items() if not v], **n})
meta = {json.loads(l)["iter"]: json.loads(l) for l in open(os.path.join(SOAK, "iterations.jsonl"))} if os.path.exists(os.path.join(SOAK, "iterations.jsonl")) else {}
for x in iters:
    m = meta.get(int(x["iter"].split("-")[1]), {})
    x.update({"rc": m.get("rc"), "start": m.get("start"), "end": m.get("end")})
verify = {}
ev_path = os.path.join(SOAK, "drumsd-events.jsonl")
n_events = 0
if os.path.exists(ev_path):
    for line in open(ev_path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "verify" in e:
            n_events += 1
            v = e["verify"]
            verify[(e.get("slot"), v.get("episode"))] = v
rewarded = [r for r in rows if r.get("reward") is not None]
bad = []
for r in rewarded:
    v = verify.get((r.get("env"), r.get("drums_episode")))
    if not v or v.get("valid") is not True or v.get("score") is None or abs(float(v["score"]) - float(r["reward"])) > 1e-9:
        bad.append({"iter": r["_iter"], "run": r["_run"], "env": r.get("env"), "episode": r.get("drums_episode"), "reward": r.get("reward"), "drumsd_verify": v})
invalid_verified = [k for k, v in verify.items() if v.get("valid") is False]
rew_invalid_slots = [r for r in rewarded if verify.get((r.get("env"), r.get("drums_episode")), {}).get("valid") is False]
inject = [json.loads(l) for l in open(os.path.join(SOAK, "inject.jsonl"))] if os.path.exists(os.path.join(SOAK, "inject.jsonl")) else []
recov = []
for ev in inject:
    if ev.get("event") not in ("drumsd_sigkill", "crashed_restore"):
        continue
    after = [r for r in rows if (r.get("t") or 0) > ev["t"]]
    first_ok = min((r["t"] for r in after if r.get("valid") and r.get("reward") is not None), default=None)
    near = [r for r in rows if abs((r.get("t") or 0) - ev["t"]) < 600]
    recov.append({**ev, "first_valid_rewarded_after_s": round(first_ok - ev["t"], 1) if first_ok else None,
                  "traces_after": len(after), "valid_rewarded_after": sum(1 for r in after if r.get("valid") and r.get("reward") is not None),
                  "errors_within_10min": dict(collections.Counter((r.get("errors") or ["?"])[0][:120] for r in near if r.get("ok") is False)),
                  "fork_children_failed_within_10min": sum(1 for r in near if r.get("fork_child_failed")),
                  "iterations_started_after": sum(1 for x in iters if (x.get("start") or 0) > ev["t"])})
manual = open(os.path.join(SOAK, "manual.log")).read().splitlines() if os.path.exists(os.path.join(SOAK, "manual.log")) else []
t0 = float(open(os.path.join(SOAK, "t0")).read()) if os.path.exists(os.path.join(SOAK, "t0")) else None
t_end = max((x.get("end") or 0 for x in iters), default=None)
tot = collections.Counter()
for x in iters:
    for k in ("traces", "valid_rewarded", "dropped", "errored", "fork_child_failed", "episodes_failed", "admission_waits", "cleanup_failed"):
        tot[k] += x.get(k, 0)
out = {"wall_minutes": round((t_end - t0) / 60, 1) if t0 and t_end else None, "iterations": len(iters),
       "iterations_passed": sum(1 for x in iters if x["pass"] is True),
       "iterations_cut_at_hard_stop": [x["iter"] for x in iters if x["partial_at_deadline"]], "totals": dict(tot),
       "rewards": dict(collections.Counter(str(r.get("reward")) for r in rewarded)),
       "invalid_as_valid": len(bad), "invalid_as_valid_rows": bad[:50],
       "drumsd_verify_events": n_events, "drumsd_verify_invalid": len(invalid_verified), "rewarded_on_drumsd_invalid": len(rew_invalid_slots),
       "recoveries": recov, "injector": inject, "manual_interventions": len(manual), "manual_log": manual,
       "per_iteration": iters}
json.dump(out, open(os.path.join(SOAK, "soak-summary.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("per_iteration", "invalid_as_valid_rows", "injector")}, indent=1, default=str))
for x in iters:
    print(x["iter"], {k: x.get(k) for k in ("rc", "pass", "traces", "valid_rewarded", "dropped", "errored", "fork_child_failed",
                                             "episodes_failed", "admission_waits", "failed_checks")})
