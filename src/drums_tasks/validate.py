"""Checker validation on a live drumsd host: every task's checker must separate its scripted oracle from its near misses.

    python3 -m drums_tasks.validate --tasks DIR --out OUT [--workers 10] [--only FAMILY,...] [--limit N] [--sample K]

Per task (standard library + the drums SDK only):
  1. create an env with the task (its setup writes the seeded files and opens the app), wait until valid;
  2. fork it into 3 + k children from that one set-up state: noop, oracle, k near misses, scope probe;
  3. in each child run its scripted actions (env.step), then verify (the reward with its audit);
  4. accept the task only if
       oracle       valid, score 1.0, audit trusted (recheck and redundant checker agree, no scope violation)
       near misses  every one valid and scored 0.0 (at least two)
       noop         valid, scored 0.0, no scope violation (the start manifest is right)
       scope probe  the audit flags scope_violation (the oracle plus one edit of a file the task never mentions)
     otherwise reject it with every reason.
Writes OUT/records.jsonl (one line per task: every child's score, validity, audit, step failures, seconds) and
OUT/summary.json. Resumable: tasks already in records.jsonl are skipped. The sample is seeded (--sample-seed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import traceback

import drums

from .upload import ensure_verifiers, sources_from_files

LOCK = threading.Lock()


def load(dirname):
    tasks = [json.loads(x) for x in open(os.path.join(dirname, "tasks.jsonl"))]
    sols = {s["id"]: s for s in (json.loads(x) for x in open(os.path.join(dirname, "solutions.jsonl")))}
    return tasks, sols


def retry_overloaded(fn, tries=60):
    """drumsd's admission control answers 429 while too much memory is promised to running operations: wait, retry."""
    for i in range(tries):
        try:
            return fn()
        except drums.Overloaded:
            if i == tries - 1:
                raise
            time.sleep(min(30, 3 + i * 2) + random.random() * 3)


def run_actions(env, actions):
    fails = []
    for i, a in enumerate(actions):
        try:
            r = retry_overloaded(lambda: env.step(a, check=False))
            res = r.result or {}
            if res.get("returncode") not in (0, None) or res.get("status") == "error":
                fails.append({"i": i, "rc": res.get("returncode"), "err": (res.get("error") or "")[-300:]})
        except drums.DrumsError as e:
            fails.append({"i": i, "exc": f"{type(e).__name__}: {e}"[:300]})
    return fails


def child_record(name, env, actions):
    t0 = time.time()
    fails = run_actions(env, actions)
    t1 = time.time()
    try:
        r = retry_overloaded(lambda: env.verify(strict=False))
        out = {"name": name, "score": r.score, "valid": r.valid, "checker_error": r.checker_error,
               "audit": r.audit, "trusted": r.trusted, "info": (r.raw.get("checker") or {}).get("info"),
               "invalid_reasons": r.invalid_reasons}
    except drums.DrumsError as e:
        out = {"name": name, "score": None, "valid": False, "error": f"{type(e).__name__}: {e}"[:400]}
    out.update(step_failures=fails, n_actions=len(actions), act_s=round(t1 - t0, 2), verify_s=round(time.time() - t1, 2))
    return out


def judge(rec):
    why = []
    kids = {k["name"]: k for k in rec.get("children", [])}
    o = kids.get("oracle")
    if rec.get("setup_error"):
        return False, ["setup: " + rec["setup_error"][:120]]
    if not o:
        return False, ["no oracle child"]
    if o.get("step_failures"):
        why.append(f"oracle step failed: {o['step_failures'][0]}")
    if not o.get("valid"):
        why.append("oracle episode invalid")
    if o.get("score") != 1.0:
        why.append(f"oracle scored {o.get('score')} ({o.get('checker_error') or (o.get('info') or {})})")
    elif not o.get("trusted"):
        why.append(f"oracle audit flags {((o.get('audit') or {}).get('flags'))}")
    nms = [k for n, k in kids.items() if n.startswith("nm:")]
    if len(nms) < 2:
        why.append("fewer than two near misses")
    for k in nms:
        if not k.get("valid"):
            why.append(f"{k['name']} invalid")
        elif k.get("score") != 0.0:
            why.append(f"{k['name']} scored {k.get('score')} {k.get('checker_error') or ''}".strip())
    n = kids.get("noop")
    if not n or n.get("score") != 0.0:
        why.append(f"noop scored {None if not n else n.get('score')}")
    elif ((n.get("audit") or {}).get("scope") or {}).get("violations"):
        why.append(f"noop scope violations {n['audit']['scope']}")
    p = kids.get("scope_probe")
    if not p or "scope_violation" not in ((p.get("audit") or {}).get("flags") or []):
        why.append("scope probe not flagged")
    return not why, why


def run_task(c, task, sol, args):
    rec = {"id": task["id"], "family": sol["family"], "app": sol["app"], "t_start": time.time()}
    parent, kids = None, []
    try:
        t0 = time.time()
        try:
            parent = retry_overloaded(lambda: c.create(task=task, idle_timeout=900))
            parent.wait_valid(timeout=120)
        except drums.DrumsError as e:
            rec["setup_error"] = f"{type(e).__name__}: {e}"[:600]
            return rec
        rec["create_s"] = round(time.time() - t0, 2)
        cands = [("noop", []), ("oracle", sol["oracle"])] + [("nm:" + m["name"], m["actions"]) for m in sol["near_misses"]] + \
                [("scope_probe", sol["scope_probe"])]
        t1 = time.time()
        fr = retry_overloaded(lambda: parent.fork(len(cands), strict=False, keep_state=False, idle_timeout=900))
        kids = list(fr)
        rec["fork_s"] = round(time.time() - t1, 2)
        rec["fork_failed"] = [str(x)[:200] for x in (fr.failed or [])]
        if len(kids) != len(cands):
            rec["setup_error"] = f"fork made {len(kids)} of {len(cands)} children"
            return rec
        with cf.ThreadPoolExecutor(len(cands)) as ex:
            rec["children"] = list(ex.map(lambda nk: child_record(nk[0][0], nk[1], nk[0][1]), zip(cands, kids)))
    except Exception as e:  # noqa: BLE001 - a harness error is recorded, not raised
        rec["setup_error"] = f"harness: {type(e).__name__}: {e}"[:600]
        rec["trace"] = traceback.format_exc()[-1500:]
    finally:
        for k in kids:
            try:
                k.close()
            except Exception:  # noqa: BLE001
                pass
        if parent is not None:
            try:
                parent.close()
            except Exception:  # noqa: BLE001
                pass
        rec["wall_s"] = round(time.time() - rec["t_start"], 2)
    return rec


def summarize(records):
    fam = {}
    for r in records:
        f = fam.setdefault(r["family"], {"tasks": 0, "accepted": 0, "app": r["app"], "reasons": {}})
        f["tasks"] += 1
        if r["accepted"]:
            f["accepted"] += 1
        for w in r["reasons"]:
            key = w.split(":")[0] if not w.startswith("nm:") else "near miss scored 1 / invalid"
            key = key if len(key) < 60 else key[:60]
            f["reasons"][key] = f["reasons"].get(key, 0) + 1
    n = len(records)
    acc = sum(r["accepted"] for r in records)
    kids = [k for r in records for k in r.get("children", [])]
    audits = [k for k in kids if k.get("audit")]
    flags = {}
    for k in audits:
        for fl in k["audit"].get("flags") or []:
            flags[fl] = flags.get(fl, 0) + 1
    return {"tasks": n, "accepted": acc, "acceptance_rate": round(acc / n, 4) if n else None,
            "near_misses_run": sum(1 for k in kids if k["name"].startswith("nm:")),
            "rollouts_verified": len(kids), "rollouts_with_audit": len(audits),
            "audit_recheck_ran": sum(1 for k in audits if (k["audit"].get("recheck") or {}).get("ran")),
            "audit_redundant_ran": sum(1 for k in audits if (k["audit"].get("redundant") or {}).get("ran")),
            "audit_scope_ran": sum(1 for k in audits if (k["audit"].get("scope") or {}).get("ran")),
            "audit_flags": flags, "families": dict(sorted(fam.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per family")
    ap.add_argument("--sample", type=int, default=0, help="K random tasks per family (seeded)")
    ap.add_argument("--sample-seed", type=int, default=20261108)
    ap.add_argument("--ids", default="", help="a file of task ids (one per line): run only these")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.drums-token"))
    ap.add_argument("--verifier", action="append", default=[],
                    help="a verifier file the tasks name by hash (uploaded if the host lacks it; the bundled drums_t6 is always known)")
    a = ap.parse_args(argv)
    tasks, sols = load(a.tasks)
    only = set(a.only.split(",")) if a.only else None
    by_fam = {}
    for t in tasks:
        by_fam.setdefault(sols[t["id"]]["family"], []).append(t)
    pick = []
    rnd = random.Random(a.sample_seed)
    for fam in sorted(by_fam):
        if only and fam not in only:
            continue
        ts = by_fam[fam]
        if a.sample:
            ts = sorted(rnd.sample(ts, min(a.sample, len(ts))), key=lambda t: t["id"])
        elif a.limit:
            ts = ts[:a.limit]
        pick += ts
    if a.ids:
        want = {x.strip() for x in open(a.ids) if x.strip()}
        pick = [t for t in pick if t["id"] in want]
    os.makedirs(a.out, exist_ok=True)
    rp = os.path.join(a.out, "records.jsonl")
    done = set()
    if os.path.exists(rp):
        done = {json.loads(x)["id"] for x in open(rp)}
    todo = [t for t in pick if t["id"] not in done]
    print(f"{len(pick)} tasks picked, {len(done)} done, {len(todo)} to run, {a.workers} workers", flush=True)
    c = drums.connect(token=open(a.token_file).read().strip())
    # tasks that name a verifier by hash need it on the host before their first create (else 422 verifier_missing)
    up = ensure_verifiers(c, todo, sources_from_files(a.verifier))
    if up:
        print("verifiers: " + ", ".join(f"{h[:12]} {s}" for h, s in sorted(up.items())), flush=True)
    t0 = time.time()
    n = [0]

    def one(t):
        rec = run_task(c, t, sols[t["id"]], a)
        rec["accepted"], rec["reasons"] = judge(rec)
        with LOCK:
            with open(rp, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            n[0] += 1
            if n[0] % 10 == 0 or not rec["accepted"]:
                print(f"[{time.time() - t0:7.0f}s] {n[0]}/{len(todo)} {rec['id']} {'ACCEPT' if rec['accepted'] else 'REJECT ' + '; '.join(rec['reasons'])[:200]}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    records = [json.loads(x) for x in open(rp)]
    keep = {t["id"] for t in pick}
    s = summarize([r for r in records if r["id"] in keep])
    s.update(wall_s=round(time.time() - t0, 1), workers=a.workers, tasks_dir_index=json.load(open(os.path.join(a.tasks, "index.json"))))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    print(json.dumps({k: s[k] for k in ("tasks", "accepted", "acceptance_rate", "audit_flags")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
