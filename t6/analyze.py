"""analyze.py RAW_DIR OUT_DIR: every table and figure of the T6 report, from raw/ only.

Reads RAW_DIR/validate-run1/records.jsonl (and validate-repro/records.jsonl, adapters/*.jsonl, hybrid/*.json,
calibration/episodes.jsonl when present). Writes OUT_DIR/tables.md, OUT_DIR/analysis.json and OUT_DIR/acceptance_by_family.svg.
"""
import collections
import json
import os
import statistics
import sys


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (round((c - r) / d, 4), round((c + r) / d, 4))


def reason_class(w):
    if w.startswith("setup: GateFailed") or "invalid" in w and "episode" in w:
        return "infra: episode invalid (gate)"
    if w.startswith("setup:"):
        return "infra: setup/fork failed"
    if "invalid" in w:
        return "infra: episode invalid (gate)"
    if w.startswith("oracle"):
        return "checker: oracle not scored 1 / not trusted"
    if w.startswith("nm:"):
        return "checker: near miss scored 1"
    if w.startswith("noop"):
        return "checker: start state scored 1 / scope"
    if w.startswith("scope probe"):
        return "audit: scope probe not flagged"
    return "other"


def run_stats(recs):
    fam = collections.OrderedDict()
    for r in sorted(recs, key=lambda r: r["id"]):
        f = fam.setdefault(r["family"], {"app": r["app"], "n": 0, "acc": 0, "classes": collections.Counter()})
        f["n"] += 1
        f["acc"] += bool(r["accepted"])
        if not r["accepted"]:
            f["classes"][reason_class(r["reasons"][0])] += 1
    n = len(recs)
    acc = sum(bool(r["accepted"]) for r in recs)
    kids = [k for r in recs for k in r.get("children", [])]
    aud = [k for k in kids if k.get("audit")]
    flags = collections.Counter(f for k in aud for f in (k["audit"].get("flags") or []))
    cls = collections.Counter(reason_class(r["reasons"][0]) for r in recs if not r["accepted"])
    infra = sum(v for k, v in cls.items() if k.startswith("infra"))
    return {"tasks": n, "accepted": acc, "rate": round(acc / n, 4) if n else None, "rate_ci95": wilson(acc, n),
            "reject_classes": dict(cls), "rate_excluding_infra_rejects": round(acc / (n - infra), 4) if n - infra else None,
            "children_verified": len(kids), "children_with_audit": len(aud),
            "audit_recheck_ran": sum(1 for k in aud if (k["audit"].get("recheck") or {}).get("ran")),
            "audit_redundant_ran": sum(1 for k in aud if (k["audit"].get("redundant") or {}).get("ran")),
            "audit_scope_ran": sum(1 for k in aud if (k["audit"].get("scope") or {}).get("ran")),
            "audit_flags": dict(flags),
            "near_misses_run": sum(1 for k in kids if k["name"].startswith("nm:")),
            "near_misses_scored_0": sum(1 for k in kids if k["name"].startswith("nm:") and k.get("score") == 0.0),
            "near_misses_scored_1": sum(1 for k in kids if k["name"].startswith("nm:") and k.get("score") == 1.0),
            "oracle_scored_1": sum(1 for k in kids if k["name"] == "oracle" and k.get("score") == 1.0),
            "redundant_disagreements": sum(1 for k in aud if "redundant_disagree" in (k["audit"].get("flags") or [])),
            "nondeterministic": sum(1 for k in aud if "nondeterministic" in (k["audit"].get("flags") or [])),
            "scope_probe_flagged": sum(1 for k in kids if k["name"] == "scope_probe" and "scope_violation" in ((k.get("audit") or {}).get("flags") or [])),
            "scope_probe_run": sum(1 for k in kids if k["name"] == "scope_probe"),
            "wall_s_p50": statistics.median([r["wall_s"] for r in recs if "wall_s" in r]) if recs else None,
            "families": {f: {"app": v["app"], "tasks": v["n"], "accepted": v["acc"], "rate": round(v["acc"] / v["n"], 3),
                             "rejects": dict(v["classes"])} for f, v in fam.items()}}


def svg(fams, path):
    names = list(fams)
    h = 18 * len(names) + 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{h}" font-family="sans-serif" font-size="11">',
           '<text x="10" y="16" font-size="13">Checker validation: accepted / generated tasks per family (run 1)</text>']
    for i, f in enumerate(names):
        y = 30 + 18 * i
        v = fams[f]
        w = 400 * v["rate"]
        out.append(f'<text x="10" y="{y + 11}">{f} (n={v["tasks"]})</text>')
        out.append(f'<rect x="230" y="{y}" width="400" height="13" fill="#e5e5e5"/><rect x="230" y="{y}" width="{w:.1f}" height="13" fill="#3b6ea5"/>')
        out.append(f'<text x="636" y="{y + 11}">{v["accepted"]}/{v["tasks"]}</text>')
    out.append(f'<text x="230" y="{h - 8}">0</text><text x="620" y="{h - 8}">1.0 (acceptance rate)</text></svg>')
    open(path, "w").write("\n".join(out))


def main():
    raw, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    res = {}
    r1 = jl(os.path.join(raw, "validate-run1", "records.jsonl"))
    res["run1"] = run_stats(r1)
    r2 = jl(os.path.join(raw, "validate-repro", "records.jsonl"))
    if r2:
        res["repro"] = run_stats(r2)
        a = {r["id"]: r["accepted"] for r in r1}
        b = {r["id"]: r["accepted"] for r in r2}
        common = sorted(set(a) & set(b))
        res["agreement"] = {"common": len(common), "same_verdict": sum(a[t] == b[t] for t in common),
                            "diff_pp": round(100 * (res["repro"]["rate"] - res["run1"]["rate"]), 2),
                            "disagree": [t for t in common if a[t] != b[t]]}
    ad = jl(os.path.join(raw, "adapters", "adapters.jsonl"))
    if ad:
        res["adapters"] = {k: {"rollouts": sum(1 for r in ad if r["adapter"] == k), "pass": sum(1 for r in ad if r["adapter"] == k and r["pass"])}
                           for k in sorted({r["adapter"] for r in ad})}
    hp = os.path.join(raw, "hybrid", "hybrid.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        res["hybrid"] = {"n": h["n"], "pass": h["pass"],
                         "download_visible_s": [x.get("download_visible_s") for x in h["results"]]}
    ce = jl(os.path.join(raw, "calibration", "episodes.jsonl"))
    if ce:
        per = collections.defaultdict(list)
        for e in ce:
            if e.get("valid") and e.get("score") is not None:
                per[e["task"]].append(e["score"])
        rates = {t: sum(v) / len(v) for t, v in per.items()}
        fam = {e["task"]: e["family"] for e in ce}
        bins = collections.Counter("0" if r == 0 else "1" if r == 1 else "mixed" for r in rates.values())
        res["calibration"] = {"episodes": len(ce), "valid_scored": sum(len(v) for v in per.values()), "tasks": len(rates),
                              "all_fail": bins["0"], "all_pass": bins["1"], "mixed": bins["mixed"],
                              "errors": sum(1 for e in ce if e.get("error")),
                              "family_mean": {f: round(statistics.mean([r for t, r in rates.items() if fam[t] == f]), 3)
                                              for f in sorted({fam[t] for t in rates})}}
    json.dump(res, open(os.path.join(out, "analysis.json"), "w"), indent=1, sort_keys=True)
    svg(res["run1"]["families"], os.path.join(out, "acceptance_by_family.svg"))
    L = ["| family | app | tasks | accepted | rate | reject classes |", "|---|---|---|---|---|---|"]
    for f, v in res["run1"]["families"].items():
        L.append(f"| {f} | {v['app']} | {v['tasks']} | {v['accepted']} | {v['rate']:.3f} | {', '.join(f'{k}: {n}' for k, n in v['rejects'].items()) or '-'} |")
    open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({k: (v if k != "run1" else {x: v[x] for x in v if x != "families"}) for k, v in res.items() if k not in ("repro",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
