"""summarize_grpo.py OUT: per-step table of a run-grpo-smoke.sh run (writes OUT/steps.json, prints a text table).

Sources, all written by the run itself:
  OUT/<run>/**/monitors/file/metrics.jsonl   prime-rl's metric lines (orchestrator and trainer), keyed by "step"
  OUT/record.jsonl                           the adapter's line per scored trace (valid, verdict, reward, drop reason, timings)
  OUT/<run>/**/checkpoints/step_*            checkpoint directories (size on disk)
Per step: every timing (time/*), reward, dropped / error / invalid counters, progress counters, and the checkpoint saved."""
import collections
import glob
import json
import os
import re
import subprocess
import sys

OUT = sys.argv[1]
KEEP = re.compile(r"^(time/|timing/|reward|env/|progress/|perf/|loss/|.*dropped|.*has_error|.*drums_valid|.*drums_actions|.*is_truncated|"
                  r".*num_turns|.*stop|.*staleness|.*mismatch_kl|.*entropy|.*grad_norm|.*advantage|.*effective|batch/|.*samples)")


def metric_files():
    return sorted(glob.glob(os.path.join(OUT, "**", "monitors", "file", "metrics.jsonl"), recursive=True))


steps = collections.defaultdict(dict)
sources = {}
for f in metric_files():
    role = "trainer" if "/trainer" in f else ("orchestrator" if "/orchestrator" in f else os.path.relpath(f, OUT))
    n = 0
    for line in open(f):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = row.get("step")
        if s is None:
            continue
        n += 1
        for k, v in row.items():
            if KEEP.match(k) and isinstance(v, (int, float)):
                steps[int(s)][f"{role}:{k}"] = v
    sources[role] = {"file": os.path.relpath(f, OUT), "lines": n}
rec = [json.loads(l) for l in open(os.path.join(OUT, "record.jsonl"))] if os.path.exists(os.path.join(OUT, "record.jsonl")) else []
records = {"traces": len(rec), "valid_rewarded": sum(1 for r in rec if r.get("valid") and r.get("reward") is not None),
           "dropped": sum(1 for r in rec if r.get("dropped")), "errored": sum(1 for r in rec if r.get("ok") is False),
           "fork_children_failed": sum(1 for r in rec if r.get("fork_child_failed")),
           "reward_on_invalid": sum(1 for r in rec if r.get("valid") is False and r.get("reward") is not None),
           "rewards": dict(collections.Counter(str(r.get("reward")) for r in rec if not r.get("fork_child_failed"))),
           "drop_reasons": dict(collections.Counter(r["dropped"] for r in rec if r.get("dropped"))),
           "errors": dict(collections.Counter((r.get("errors") or ["?"])[0][:160] for r in rec if r.get("ok") is False)),
           "fork_groups": len({r.get("parent") for r in rec if r.get("parent")}),
           "agent_s_sum": round(sum(r.get("agent_s") or 0 for r in rec), 1),
           "ready_s_mean": round(sum(r.get("ready_s") or 0 for r in rec) / max(1, len(rec)), 3),
           "fork_s_mean": round(sum(r.get("fork_s") or 0 for r in rec) / max(1, len(rec)), 3)}
ckpts = []
for d in sorted(glob.glob(os.path.join(OUT, "**", "checkpoints", "step_*"), recursive=True)):
    size = subprocess.run(["du", "-sb", d], capture_output=True, text=True).stdout.split()[:1]
    ckpts.append({"dir": os.path.relpath(d, OUT), "bytes": int(size[0]) if size else None})
out = {"steps": {str(k): steps[k] for k in sorted(steps)}, "metric_sources": sources, "records": records, "checkpoints": ckpts}
json.dump(out, open(os.path.join(OUT, "steps.json"), "w"), indent=1)
cols = ["time/step", "time/wait_for_batch", "time/forward_backward", "time/broadcast_weights", "time/save_ckpt", "reward"]
for k in sorted(steps):
    row = steps[k]
    short = {key: (round(v, 3) if isinstance(v, float) else v) for key, v in row.items() if any(c in key for c in cols) or "dropped" in key or "error" in key}
    print(f"step {k}: {json.dumps(short)}")
print("records:", json.dumps(records))
print("checkpoints:", json.dumps(ckpts))
