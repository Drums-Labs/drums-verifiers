"""compare_runs.py RUN1/records.jsonl RUN2/records.jsonl: per-task accept/reject agreement between two validation runs."""
import json
import sys


def load(p):
    return {r["id"]: r for r in (json.loads(x) for x in open(p))}


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
agree = [t for t in common if a[t]["accepted"] == b[t]["accepted"]]
dis = [t for t in common if a[t]["accepted"] != b[t]["accepted"]]
ra = sum(a[t]["accepted"] for t in common) / len(common)
rb = sum(b[t]["accepted"] for t in common) / len(common)
out = {"tasks_common": len(common), "only_run1": len(set(a) - set(b)), "only_run2": len(set(b) - set(a)),
       "acceptance_run1": round(ra, 4), "acceptance_run2": round(rb, 4), "diff_pp": round(100 * (rb - ra), 2),
       "verdict_agreement": round(len(agree) / len(common), 4),
       "disagreements": [{"id": t, "run1": a[t]["accepted"], "run2": b[t]["accepted"],
                          "run1_reasons": a[t]["reasons"][:2], "run2_reasons": b[t]["reasons"][:2]} for t in dis]}
print(json.dumps(out, indent=1))
