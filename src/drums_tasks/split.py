"""Training and held-out variants of one family from two seeds, with an overlap check:

    python -m drums_tasks.split --family NAME --train-seed A --heldout-seed B --out DIR [--train 64] [--heldout 32]
                                [--start 0]

Writes, in DIR:
  train/      a full task set (tasks.jsonl, solutions.jsonl, drums_t6.py, index.json): indices start..start+train-1
              under seed A
  heldout/    the same for the next `heldout` indices under seed B (so no task id appears in both sets)
  pool.json   family, app, generator version, both seeds and index ranges, the sha256 of each set's files, every task id
              with its split and params, and the overlap report

A held-out variant overlaps a training variant when it has the same instruction text, a seeded file with the same path
and content, or the same params dict. Any overlap fails the split (exit 1; pool.json still lists the collisions).
Each set is a normal task set: `drums_tasks.validate --tasks DIR/train` proves it on a host. `--start` moves both index
ranges, so a later pool of a family never reuses the task ids (and the id-bearing decoy file) of an earlier pool.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys

from . import GENERATOR_VERSION
from .build import ALL, write
from .core import _WRITE, canonical


def seeded_files(task: dict) -> dict:
    """{path: sha256 of the content} of every file the task's setup writes (decoded from the chunked write steps)."""
    chunks = {}
    for st in task["config"]:
        if st.get("type") != "execute":
            continue
        cmd = st["parameters"].get("command") or []
        if len(cmd) == 6 and cmd[0] == "python3" and cmd[2] == _WRITE:
            path, mode, b64 = cmd[3], cmd[4], cmd[5]
            if mode == "wb":
                chunks[path] = [b64]
            else:
                chunks.setdefault(path, []).append(b64)
    return {p: hashlib.sha256(base64.b64decode("".join(c))).hexdigest() for p, c in chunks.items()}


def overlaps(train: list, heldout: list) -> list:
    """train, heldout: lists of (task, solution). -> one record per collision (held-out id, training id, what)."""
    by_instr = {t["instruction"]: t["id"] for t, _ in train}
    by_file = {(p, h): t["id"] for t, _ in train for p, h in seeded_files(t).items()}
    by_params = {canonical(s["params"]): t["id"] for t, s in train}
    out = []
    for t, s in heldout:
        if t["instruction"] in by_instr:
            out.append({"heldout": t["id"], "train": by_instr[t["instruction"]], "kind": "instruction"})
        for p, h in seeded_files(t).items():
            if (p, h) in by_file:
                out.append({"heldout": t["id"], "train": by_file[(p, h)], "kind": "file", "path": p})
        key = canonical(s["params"])
        if key in by_params:
            out.append({"heldout": t["id"], "train": by_params[key], "kind": "params"})
    return out


def _load(d):
    tasks = [json.loads(x) for x in open(os.path.join(d, "tasks.jsonl"))]
    sols = [json.loads(x) for x in open(os.path.join(d, "solutions.jsonl"))]
    return list(zip(tasks, sols))


def split(family: str, train_seed: int, heldout_seed: int, out: str, n_train: int = 64, n_heldout: int = 32,
          start: int = 0) -> dict:
    if family not in ALL:
        raise SystemExit(f"unknown family {family!r}; known: {', '.join(sorted(ALL))}")
    sets = {}
    for name, seed, first, n in (("train", train_seed, start, n_train),
                                 ("heldout", heldout_seed, start + n_train, n_heldout)):
        d = os.path.join(out, name)
        idx = write(d, seed, 1.0, {family}, count=n, start=first)
        sets[name] = {"seed": seed, "start": first, "count": n, "dir": name, "sha256": idx["sha256"], "records": _load(d)}
    coll = overlaps(sets["train"]["records"], sets["heldout"]["records"])
    app = sets["train"]["records"][0][1]["app"]
    pool = {"family": family, "app": app, "generator_version": GENERATOR_VERSION,
            "train": {k: v for k, v in sets["train"].items() if k != "records"},
            "heldout": {k: v for k, v in sets["heldout"].items() if k != "records"},
            "variants": [{"id": t["id"], "split": name, "params": s["params"]}
                         for name in ("train", "heldout") for t, s in sets[name]["records"]],
            "overlap": {"checks": ["instruction", "file", "params"], "collisions": coll, "ok": not coll}}
    for name in ("train", "heldout"):
        pool[name]["ids"] = [t["id"] for t, _ in sets[name]["records"]]
    with open(os.path.join(out, "pool.json"), "w") as f:
        json.dump(pool, f, indent=1, sort_keys=True)
        f.write("\n")
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--heldout-seed", type=int, required=True)
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--heldout", type=int, default=32)
    ap.add_argument("--start", type=int, default=0, help="the first training index (held-out indices follow)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.train_seed == a.heldout_seed:
        print("the two seeds must differ", file=sys.stderr)
        return 2
    pool = split(a.family, a.train_seed, a.heldout_seed, a.out, a.train, a.heldout, a.start)
    summary = {k: pool[k] for k in ("family", "app", "generator_version")}
    summary.update({name: {k: pool[name][k] for k in ("seed", "start", "count", "sha256")} for name in ("train", "heldout")},
                   collisions=len(pool["overlap"]["collisions"]))
    print(json.dumps(summary, indent=1))
    if not pool["overlap"]["ok"]:
        print(f"OVERLAP: {len(pool['overlap']['collisions'])} collision(s), see pool.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
