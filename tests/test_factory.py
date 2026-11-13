"""The task factory and its verifier, without a desktop: deterministic builds, task shape, and the checkers on
files written the way the scripted solutions write them (a stand-in ctx that maps desktop paths into a temp dir).

    .venv-factory/bin/python -m pytest tests/test_factory.py      (needs the generator pins: t6/requirements-factory.txt)
"""
import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
pytest.importorskip("openpyxl")
pytest.importorskip("docx")
pytest.importorskip("pptx")

from drums_tasks import build  # noqa: E402

spec = importlib.util.spec_from_file_location("drums_t6", os.path.join(HERE, "..", "src", "drums_tasks", "verifier", "drums_t6.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


@pytest.fixture(scope="module")
def small():
    return build.generate(20261108, scale=0.05)


def test_same_seed_same_bytes_and_other_seed_differs(small):
    t2, s2 = build.generate(20261108, scale=0.05)
    from drums_tasks.core import canonical
    assert [canonical(x) for x in small[0]] == [canonical(x) for x in t2]
    assert [canonical(x) for x in small[1]] == [canonical(x) for x in s2]
    t3, _ = build.generate(7, scale=0.05)
    assert [canonical(x) for x in t3] != [canonical(x) for x in small[0]]


def test_every_task_is_well_formed_and_keeps_its_solutions_out(small):
    tasks, sols = small
    assert len({t["id"] for t in tasks}) == len(tasks) == len(sols)
    for t, s in zip(tasks, sols):
        assert t["id"] == s["id"]
        ev = t["evaluator"]
        assert ev["func"] == "drums_verifier" and ev["verifier"] == "drums_t6" and ev["options"]["kind"]
        assert len(s["near_misses"]) >= 2, t["id"]
        assert s["oracle"] and s["scope_probe"]
        blob = json.dumps(t)
        assert "near_miss" not in blob and '"oracle"' not in blob          # no solution inside the task an agent runs
        assert all(len(a) < 128 * 1024 for st in t["config"] for a in st["parameters"].get("command", []) if isinstance(a, str))
        man = ev["scope"]["manifest"]
        assert "/home/user/Documents/README-keep.txt" in man                  # the decoy the scope probe edits


def test_full_set_counts():
    n = sum(n for _, n in build.ALL.values())
    assert n >= 1000
    assert len(build.ALL) >= 30


class Ctx:
    """drumsd's checker ctx, backed by a local directory standing in for the desktop's filesystem."""

    def __init__(self, root):
        self.root, self.workdir, self.options, self.task = root, root, {}, {}

    def p(self, path):
        return os.path.join(self.root, path.lstrip("/"))

    def read_file(self, path):
        q = self.p(path)
        if not os.path.isfile(q):
            return None
        return open(q, "rb").read()

    def run(self, command, shell=False, timeout=None):
        cmd = [c.replace("/home/user", os.path.join(self.root, "home/user")) if isinstance(c, str) else c for c in command]
        if cmd[0] == "timeout":
            cmd = cmd[2:]
        if cmd[0] == "python3":
            cmd[0] = sys.executable
        r = subprocess.run(cmd, capture_output=True, text=True)
        return {"returncode": r.returncode, "output": r.stdout, "error": r.stderr}


def apply(ctx, actions):
    """Run a solution's write_file steps (the python3 -c writer) and plain shell steps on the stand-in filesystem."""
    for a in actions:
        if a["type"] != "execute":
            continue
        ctx.run(a["command"])


def _family_cases(sols, tasks, fam):
    return [(t, s) for t, s in zip(tasks, sols) if s["family"] == fam]


@pytest.mark.parametrize("fam", ["files_rename", "files_move_ext", "files_count", "files_concat", "files_find_copy",
                                 "files_mkdir_tree", "files_delete_pattern", "vscode_setting", "vscode_fix_function",
                                 "vscode_rename_symbol", "gimp_transform", "gimp_resize_crop", "multi_calc_to_text",
                                 "multi_web_to_csv"])
def test_checker_separates_oracle_from_near_misses_offline(small, fam):
    """Families whose solutions are plain file writes or shell steps run here end to end (no LibreOffice needed)."""
    tasks, sols = small
    cases = _family_cases(sols, tasks, fam)
    assert cases
    for t, s in cases:
        for name, actions, want in [("oracle", s["oracle"], 1.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]:
            with tempfile.TemporaryDirectory() as d:
                ctx = Ctx(d)
                for st in t["config"]:
                    if st["type"] == "execute":
                        ctx.run(st["parameters"]["command"])
                apply(ctx, actions)
                opts = dict(t["evaluator"]["options"])
                kind = opts.pop("kind")
                r1 = V.check(ctx, kind, **dict(opts))
                assert r1["score"] == want, (t["id"], name, r1)
                if kind not in ("chrome_tabs",):
                    r2 = V.check_alt(ctx, kind, **dict(opts))
                    assert r2["score"] == want, (t["id"], name, "alt", r2)


def test_checkers_never_raise_on_garbage():
    with tempfile.TemporaryDirectory() as d:
        ctx = Ctx(d)
        os.makedirs(os.path.join(d, "home/user"), exist_ok=True)
        open(os.path.join(d, "home/user/x"), "wb").write(b"\x00garbage\xff" * 10)
        for kind, o in [("xlsx_table", {"path": "/home/user/x", "rows": [[1]]}), ("docx_paragraphs", {"path": "/home/user/x", "paragraphs": []}),
                        ("pptx_titles", {"path": "/home/user/x", "titles": []}), ("pdf_text", {"path": "/home/user/x", "contains": ["a"]}),
                        ("image", {"path": "/home/user/x", "expected_b64": base64.b64encode(b"x").decode()}),
                        ("csv_table", {"path": "/home/user/x", "rows": [["a"]]}), ("json_keys", {"path": "/home/user/x", "keys": {"a": 1}}),
                        ("zip_members", {"path": "/home/user/x", "members": {}}), ("text_file", {"path": "/home/user/nope", "expected": "a"})]:
            assert V.check(ctx, kind, **dict(o))["score"] == 0.0, kind
            assert V.check_alt(ctx, kind, **dict(o))["score"] == 0.0, kind
