"""The D1 study families and the training / held-out split, without a desktop.

What runs here: variants differ across indices; the checker (primary and redundant) scores the oracle 1.0 and every
near miss 0.0 on the files the solutions write; the split's overlap check rejects a duplicated variant; two builds of
the three families are byte-identical. The office oracles go through `office_write`, whose LibreOffice conversion is
replaced by a copy here (the staged file already has the target format); a sample runs through a real `soffice` when
one is installed. What still needs a live host (`drums_tasks.validate`): the GUI's own save path, the no-op child, the
scope probe and the reward audit.
"""
import base64
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
pytest.importorskip("openpyxl")
pytest.importorskip("docx")

from test_factory import Ctx, V  # noqa: E402

from drums_tasks import build, split  # noqa: E402
from drums_tasks.core import _WRITE, canonical  # noqa: E402

D1 = ["d1_files_collect", "d1_calc_aggregate", "d1_writer_style"]
SOFFICE = shutil.which("soffice")


def cases(fam, seed=20261116, count=None, start=0):
    tasks, sols = build.generate(seed, only={fam}, count=count, start=start)
    return list(zip(tasks, sols))


class OfficeCtx(Ctx):
    """The stand-in desktop also maps the oracles' /tmp staging folders into the temp root and converts office files by
    copying them (or through soffice when asked)."""

    def __init__(self, root, soffice=False):
        super().__init__(root)
        self.soffice = soffice

    def run(self, command, shell=False, timeout=None):
        cmd = [c.replace("/tmp/t6-", os.path.join(self.root, "tmp/t6-")) if isinstance(c, str) else c for c in command]
        if len(cmd) == 6 and cmd[0] == "python3" and cmd[2] == _WRITE:
            # the chunked file writer, in process (one interpreter per seeded file is what makes 96 variants slow)
            q = self.p(cmd[3].replace(self.root, "")) if cmd[3].startswith(self.root) else self.p(cmd[3])
            os.makedirs(os.path.dirname(q), exist_ok=True)
            with open(q, cmd[4]) as f:
                f.write(base64.b64decode(cmd[5]))
            return {"returncode": 0, "output": "", "error": ""}
        if cmd[:3] == ["python3", "-I", "-c"]:
            # the verifier's own snippets (the folder listing, the redundant reader), run in process with the same argv:
            # one interpreter start per read is what makes 96 variants times 5 solutions slow
            out, argv, rc = io.StringIO(), sys.argv, 0
            sys.argv = ["-c", *[self.p(c) if c.startswith("/home/user") else c for c in cmd[4:]]]
            try:
                with contextlib.redirect_stdout(out):
                    exec(cmd[3], {"__name__": "__main__"})
            except SystemExit as e:
                rc = int(e.code or 0)
            finally:
                sys.argv = argv
            return {"returncode": rc, "output": out.getvalue(), "error": ""}
        if cmd[0] == "soffice" and not self.soffice:
            fmt, outdir, src = cmd[cmd.index("--convert-to") + 1], cmd[cmd.index("--outdir") + 1], cmd[-1]
            os.makedirs(outdir, exist_ok=True)
            shutil.copy(src, os.path.join(outdir, os.path.basename(src).rsplit(".", 1)[0] + "." + fmt))
            return {"returncode": 0, "output": "", "error": ""}
        if cmd[0] == "soffice":
            cmd[0] = SOFFICE
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return {"returncode": r.returncode, "output": r.stdout, "error": r.stderr}
        return super().run(cmd, shell, timeout)


def score(task, actions, soffice=False):
    with tempfile.TemporaryDirectory() as d:
        ctx = OfficeCtx(d, soffice)
        for st in task["config"]:
            if st["type"] == "execute":
                ctx.run(st["parameters"]["command"])
        for a in actions:
            if a["type"] == "execute":
                r = ctx.run(a["command"])
                assert r["returncode"] == 0, (task["id"], a["command"][:3], r["error"][-300:])
        opts = dict(task["evaluator"]["options"])
        kind = opts.pop("kind")
        return V.check(ctx, kind, **dict(opts)), V.check_alt(ctx, kind, **dict(opts))


@pytest.mark.parametrize("fam", D1)
def test_variants_differ_across_indices(fam):
    cs = cases(fam)
    assert len(cs) == 96 and build.ALL[fam][1] >= 96
    assert len({t["instruction"] for t, _ in cs}) == 96
    assert len({canonical(s["params"]) for _, s in cs}) == 96
    assert len({canonical(sorted(split.seeded_files(t).items())) for t, _ in cs}) == 96
    for t, s in cs:
        assert len(s["near_misses"]) >= 3, t["id"]
        assert len({m["name"] for m in s["near_misses"]}) == len(s["near_misses"])


@pytest.mark.parametrize("fam", D1)
def test_oracle_passes_and_near_misses_fail(fam):
    for t, s in cases(fam):
        for name, actions, want in [("oracle", s["oracle"], 1.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]:
            r1, r2 = score(t, actions)
            assert r1["score"] == want, (t["id"], name, r1)
            assert r2["score"] == want, (t["id"], name, "alt", r2)


@pytest.mark.skipif(not SOFFICE, reason="soffice not installed")
@pytest.mark.parametrize("fam", ["d1_calc_aggregate", "d1_writer_style"])
def test_office_oracles_through_libreoffice(fam):
    """The oracle's file written by LibreOffice itself (the same writer a GUI save goes through) still separates."""
    for t, s in cases(fam, count=2):
        for name, actions, want in [("oracle", s["oracle"], 1.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]:
            r1, r2 = score(t, actions, soffice=True)
            assert r1["score"] == want, (t["id"], name, r1)
            assert r2["score"] == want, (t["id"], name, "alt", r2)


def test_overlap_rejects_a_duplicated_variant():
    train = cases("d1_files_collect", seed=1, count=4)
    held = cases("d1_files_collect", seed=2, count=2, start=4)
    assert split.overlaps(train, held) == []
    # the same indices under the other seed repeat the task ids, so the decoy file (which carries the id) collides:
    # that is why the split starts the held-out indices after the training ones
    assert {c["kind"] for c in split.overlaps(train, cases("d1_files_collect", seed=2, count=2))} == {"file"}
    dup_t, dup_s = json.loads(json.dumps(train[1][0])), json.loads(json.dumps(train[1][1]))
    dup_t["id"] = dup_s["id"] = "t6-d1-files-collect-9999"
    kinds = {c["kind"] for c in split.overlaps(train, held + [(dup_t, dup_s)])}
    assert kinds == {"instruction", "file", "params"}
    # the decoy file carries the task id, so a same-path decoy with other content is not a collision
    assert all(c["heldout"] == "t6-d1-files-collect-9999" for c in split.overlaps(train, held + [(dup_t, dup_s)]))


def test_split_writes_both_sets_and_the_pool(tmp_path):
    pool = split.split("d1_writer_style", 11, 12, str(tmp_path), 6, 3)
    assert pool["overlap"]["ok"] and len(pool["variants"]) == 9
    assert pool["train"]["ids"] == [f"t6-d1-writer-style-{i:04d}" for i in range(6)]
    assert pool["heldout"]["ids"] == [f"t6-d1-writer-style-{i:04d}" for i in range(6, 9)]
    for name in ("train", "heldout"):
        d = tmp_path / name
        import hashlib
        assert hashlib.sha256((d / "tasks.jsonl").read_bytes()).hexdigest() == pool[name]["sha256"]["tasks.jsonl"]
        idx = json.loads((d / "index.json").read_text())
        assert idx["families"] == {"d1_writer_style": pool[name]["count"]} and idx["seed"] == pool[name]["seed"]
    assert json.loads((tmp_path / "pool.json").read_text())["family"] == "d1_writer_style"
    assert split.main(["--family", "d1_writer_style", "--train-seed", "5", "--heldout-seed", "5", "--out", str(tmp_path / "x")]) == 2


def test_two_builds_are_byte_identical(tmp_path):
    args = ["--families", ",".join(D1), "--seed", "1"]
    assert build.main(args + ["--out", str(tmp_path / "a"), "--check"]) == 0
    assert build.main(args + ["--out", str(tmp_path / "b")]) == 0
    for name in ("tasks.jsonl", "solutions.jsonl", "drums_t6.py", "index.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    idx = json.loads((tmp_path / "a" / "index.json").read_text())
    assert idx["families"] == {f: 96 for f in D1}
