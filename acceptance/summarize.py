"""summarize.py RD: the acceptance verdict from the raw records (writes RD/summary.json, prints a short text form).

Per run: traces, valid / dropped / errored, rewards handed out, drop reasons, fork structure, and for every trace the
screenshots the model received (from prime-rl's own trace stream). Pass criteria are listed in acceptance.sh."""
import base64
import collections
import glob
import hashlib
import json
import os
import sys

RD = sys.argv[1]


def traces_of(run_dir):
    """Every trace of the run from prime-rl's file monitor stream (zstd JSONL of episodes)."""
    import pyzstd
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "out", "*", "monitors", "file", "traces", "stream", "*.jsonl.zst"))):
        for line in pyzstd.decompress(open(f, "rb").read()).decode().splitlines():
            if line.strip():
                ep = json.loads(line)
                for t in ep.get("traces", []):
                    out.append((ep, t))
    return out


def images(trace):
    """(role, sha256, bytes) of every image the model was given, in order."""
    out = []
    for n in trace.get("nodes", []):
        m = n.get("message") or {}
        c = m.get("content")
        for p in c if isinstance(c, list) else []:
            url = ((p.get("image_url") or {}).get("url") or "") if isinstance(p, dict) else ""
            if url.startswith("data:image"):
                b = base64.b64decode(url.split(",", 1)[1])
                out.append((m.get("role"), hashlib.sha256(b).hexdigest()[:16], len(b)))
    return out


def run_summary(name):
    d = os.path.join(RD, "runs", name)
    rec = [json.loads(l) for l in open(os.path.join(d, "record.jsonl"))] if os.path.exists(os.path.join(d, "record.jsonl")) else []
    tr = traces_of(d) if os.path.isdir(os.path.join(d, "out")) else []
    shots = []
    for ep, t in tr:
        im = images(t)
        tool_calls = sum(len((n.get("message") or {}).get("tool_calls") or []) for n in t.get("nodes", []))
        shots.append({"env": (t.get("info") or {}).get("drums", {}).get("env"), "first_user_image": bool(im and im[0][0] == "user"),
                      "tool_images": sum(1 for r, _, _ in im if r == "tool"), "tool_calls": tool_calls,
                      "distinct_images": len({h for _, h, _ in im}), "ok": t.get("ok"), "dropped": t.get("dropped")})
    s = {"traces_recorded": len(rec), "traces_in_prime_rl_stream": len(tr),
         "valid_rewarded": sum(1 for r in rec if r.get("valid") and r.get("reward") is not None),
         "dropped": sum(1 for r in rec if r.get("dropped")), "errored": sum(1 for r in rec if r.get("ok") is False),
         "fork_children_failed": sum(1 for r in rec if r.get("fork_child_failed")),
         "reward_on_invalid": sum(1 for r in rec if r.get("valid") is False and r.get("reward") is not None),
         "rewards": dict(collections.Counter(str(r.get("reward")) for r in rec if not r.get("fork_child_failed"))),
         "drop_reasons": dict(collections.Counter(r["dropped"] for r in rec if r.get("dropped"))),
         "errors": dict(collections.Counter((r.get("errors") or ["?"])[0][:160] for r in rec if r.get("ok") is False)),
         "by_task": {}, "screenshots": shots}
    for r in rec:
        k = s["by_task"].setdefault((r.get("task") or "?")[:8], {"n": 0, "reward_sum": 0.0, "valid": 0})
        k["n"] += 1
        k["valid"] += bool(r.get("valid"))
        k["reward_sum"] += r.get("reward") or 0.0
    parents = collections.defaultdict(set)
    for r in rec:
        if r.get("parent"):
            parents[r["parent"]].add(r.get("env"))
    s["fork_groups"] = {p: len(c) for p, c in parents.items()}
    for f in glob.glob(os.path.join(d, "out", "*", "monitors", "file", "metrics.jsonl")):
        last = {}
        for line in open(f):
            last.update(json.loads(line))
        s["prime_rl_metrics"] = {k: v for k, v in last.items() if any(x in k for x in ("reward/mean", "dropped", "has_error/mean", "drums_valid/mean", "pass@"))}
    return s


runs = {os.path.basename(p): run_summary(os.path.basename(p)) for p in sorted(glob.glob(os.path.join(RD, "runs", "*")))}
checks = {}
for name, s in runs.items():
    n = s["traces_recorded"]
    checks[f"{name}: traces recorded"] = n > 0
    checks[f"{name}: no reward on an invalid episode"] = s["reward_on_invalid"] == 0
    checks[f"{name}: every trace valid+rewarded, dropped, or errored"] = s["valid_rewarded"] + s["dropped"] + s["errored"] + s["fork_children_failed"] == n
    checks[f"{name}: every trace saw a first screenshot"] = all(x["first_user_image"] for x in s["screenshots"]) and bool(s["screenshots"])
    checks[f"{name}: a screenshot answered every tool call"] = all(x["tool_images"] >= x["tool_calls"] for x in s["screenshots"] if x["ok"] and not x["dropped"])
if "B-fork" in runs:
    checks["B-fork: children are distinct envs grouped under one parent each"] = bool(runs["B-fork"]["fork_groups"]) and all(v >= 1 for v in runs["B-fork"]["fork_groups"].values())
if "C-control" in runs:
    c = runs["C-control"]
    checks["C-control: every child dropped as drums_invalid (alive gate), none rewarded"] = c["traces_recorded"] > 0 and c["dropped"] == c["traces_recorded"] and all(k.startswith("drums_invalid") for k in c["drop_reasons"])
verdict = {"pass": all(checks.values()), "checks": checks}
out = {"verdict": verdict, "runs": runs}
for f in ("versions.txt", "phases.jsonl", "host.txt"):
    p = os.path.join(RD, f)
    if os.path.exists(p):
        out[f] = open(p).read()
json.dump(out, open(os.path.join(RD, "summary.json"), "w"), indent=1, default=str)
print(json.dumps(verdict, indent=1))
for name, s in runs.items():
    print(name, {k: s[k] for k in ("traces_recorded", "valid_rewarded", "dropped", "errored", "rewards", "drop_reasons", "fork_groups")})
