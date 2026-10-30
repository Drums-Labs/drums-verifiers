"""refused_calls.py RD...: for every acceptance run dir, the tool calls that got no screenshot, split into
  refused   the adapter rejected the call as an invalid action (its result is the text "Invalid action: ..."), so nothing ran
            on the desktop: no screenshot is owed. Their count per trace must equal the adapter's invalid_actions.
  other     anything else: a tool call with no screenshot that was not a refused invalid action (must be 0)
and the verdict of the as-run check set with the one rule "a screenshot answered every tool call" replaced by
"a screenshot answered every tool call except the ones the adapter refused as invalid (count = invalid_actions)".
The as-run summary.json is not changed; this writes refused_calls.json next to it."""
import glob
import json
import os
import sys

import pyzstd


def check(rd):
    out = {"rd": rd, "traces": 0, "refused": 0, "other": [], "refused_count_mismatch": [], "examples": []}
    for f in sorted(glob.glob(os.path.join(rd, "runs", "*", "out", "*", "monitors", "file", "traces", "stream", "*.jsonl.zst"))):
        run = f.split("/runs/")[1].split("/")[0]
        for line in pyzstd.decompress(open(f, "rb").read()).decode().splitlines():
            ep = json.loads(line)
            for t in ep.get("traces", []):
                if not t.get("ok") or t.get("dropped"):
                    continue
                out["traces"] += 1
                d = (t.get("info") or {}).get("drums") or {}
                calls, first = {}, {}
                for n in t.get("nodes", []):
                    m = n.get("message") or {}
                    for c in m.get("tool_calls") or []:
                        calls[c["id"]] = c
                    if m.get("role") == "tool" and m.get("tool_call_id") not in first:
                        first[m["tool_call_id"]] = m.get("content")
                refused = 0
                for cid, cont in first.items():
                    img = isinstance(cont, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in cont)
                    if img:
                        continue
                    text = cont if isinstance(cont, str) else json.dumps(cont)
                    if text.startswith("Invalid action"):
                        refused += 1
                        if len(out["examples"]) < 10:
                            out["examples"].append({"run": run, "env": d.get("env"), "call": (calls.get(cid) or {}).get("arguments"), "result": text[:120]})
                    else:
                        out["other"].append({"run": run, "env": d.get("env"), "call": (calls.get(cid) or {}).get("arguments"), "result": text[:200]})
                out["refused"] += refused
                if refused != (d.get("invalid_actions") or 0):
                    out["refused_count_mismatch"].append({"run": run, "env": d.get("env"), "refused": refused, "invalid_actions": d.get("invalid_actions")})
    s = json.load(open(os.path.join(rd, "summary.json")))
    checks = dict(s["verdict"]["checks"])
    replaced = [k for k in checks if k.endswith("a screenshot answered every tool call")]
    ok_rule = not out["other"] and not out["refused_count_mismatch"]
    for k in replaced:
        checks[k] = ok_rule
    out["as_run_pass"] = s["verdict"]["pass"]
    out["pass_with_refused_exempt"] = all(checks.values()) and not s["verdict"].get("partial_at_deadline")
    out["still_failing"] = [k for k, v in checks.items() if not v]
    json.dump(out, open(os.path.join(rd, "refused_calls.json"), "w"), indent=1)
    return out


for rd in sys.argv[1:]:
    o = check(rd)
    print(os.path.basename(rd.rstrip("/")), {k: o[k] for k in ("traces", "refused", "as_run_pass", "pass_with_refused_exempt", "still_failing")},
          "other:", len(o["other"]), "mismatch:", len(o["refused_count_mismatch"]))
