"""Build the task set: python -m drums_tasks.build --seed 20261108 --out DIR [--scale 1.0] [--families a,b]

Writes, in DIR:
  tasks.jsonl       one drumsd task JSON per line (what an agent's env is created with; no solutions inside)
  solutions.jsonl   one record per task: family, params, the scripted oracle, the near-miss solutions, the scope probe
  drums_t6.py       the lab verifier the tasks name (drumsd >= 0.2.2 ships it; older hosts: install as root into /etc/drums/verifiers/)
  index.json        counts per family and app, the seed, the generator version, sha256 of every file above, and
                    `verifier_sha256` ({name: sha256} of the verifier the tasks name)

--verifier-ref hash also names the verifier by its sha256 in every task (`evaluator.verifier_sha256`): drumsd then runs
it in its verifier sandbox and refuses a task whose hash it does not have (validate and DrumsEnv upload it). The
default, `name`, writes the same bytes as before.

The same seed and generator version give byte-identical files (checked by `--check`, which builds twice in memory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

from . import GENERATOR_VERSION
from .core import VERIFIER, build_task, canonical, rng_for
from .upload import bundled_sources, name_by_hash
from .families import apps, calc, d1, files, multi, writer

ALL = {}
for mod in (calc, writer, files, apps, multi, d1):
    ALL.update(mod.FAMILIES)


def generate(seed: int, scale: float = 1.0, only=None, count=None, start=0, verifier_ref: str = "name"):
    """count: exact variants per family instead of the registered count times scale; start: the first index."""
    tasks, sols = [], []
    for fam in sorted(ALL):
        if only and fam not in only:
            continue
        fn, n = ALL[fam]
        n = count if count is not None else max(1, int(round(n * scale)))
        for i in range(start, start + n):
            spec = fn(rng_for(seed, fam, i), i)
            tid = f"t6-{fam.replace('_', '-')}-{i:04d}"
            t, s = build_task(spec, tid, seed)
            if verifier_ref == "hash":
                t = name_by_hash(t, bundled_sources())
            tasks.append(t)
            sols.append(s)
    return tasks, sols


def write(out: str, seed: int, scale: float, only=None, count=None, start=0, verifier_ref: str = "name") -> dict:
    if verifier_ref not in ("name", "hash"):
        raise ValueError(f"verifier_ref {verifier_ref!r}: name or hash")
    os.makedirs(out, exist_ok=True)
    tasks, sols = generate(seed, scale, only, count, start, verifier_ref)
    files = {"tasks.jsonl": b"".join(canonical(t) + b"\n" for t in tasks),
             "solutions.jsonl": b"".join(canonical(s) + b"\n" for s in sols)}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "verifier", "drums_t6.py"), "rb") as f:
        files["drums_t6.py"] = f.read()
    for name, data in files.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
    fam, app = {}, {}
    for s in sols:
        fam[s["family"]] = fam.get(s["family"], 0) + 1
        app[s["app"]] = app.get(s["app"], 0) + 1
    index = {"generator_version": GENERATOR_VERSION, "seed": seed, "scale": scale, "tasks": len(tasks),
             "near_misses": sum(len(s["near_misses"]) for s in sols), "families": fam, "apps": app,
             "multi_app": sum(1 for t in tasks if "multi" in t["related_apps"]),
             "sha256": {n: hashlib.sha256(d).hexdigest() for n, d in sorted(files.items())},
             "verifier_sha256": {VERIFIER: hashlib.sha256(files["drums_t6.py"]).hexdigest()}, "verifier_ref": verifier_ref,
             "phrasing": "templates (no language model: none configured for this program)"}
    if count is not None or start:
        index.update(count=count, start=start)
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
        f.write("\n")
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--families", default="")
    ap.add_argument("--check", action="store_true", help="build twice and fail on any byte difference")
    ap.add_argument("--verifier-ref", choices=("name", "hash"), default="name",
                    help="hash: tasks also name the verifier by sha256 (run in drumsd's verifier sandbox)")
    a = ap.parse_args(argv)
    only = set(a.families.split(",")) if a.families else None
    idx = write(a.out, a.seed, a.scale, only, verifier_ref=a.verifier_ref)
    if a.check:
        tmp = a.out.rstrip("/") + ".check"
        idx2 = write(tmp, a.seed, a.scale, only, verifier_ref=a.verifier_ref)
        shutil.rmtree(tmp)
        if idx2["sha256"] != idx["sha256"]:
            print("NOT deterministic:", idx["sha256"], idx2["sha256"], file=sys.stderr)
            return 1
    print(json.dumps({k: idx[k] for k in ("tasks", "near_misses", "multi_app", "sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
