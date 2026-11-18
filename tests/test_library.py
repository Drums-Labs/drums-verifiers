"""The task library (docs/task-library.md), without a desktop: the medium, multi-app, hidden-state and recovery
families, and the lossless `params` of the pre-D1 families the audit keeps.

What runs here: variants differ across indices; the checker (primary and redundant) scores the oracle 1.0, every near
miss 0.0 and a no-op 0.0 (the task is not already solved at the start, which matters most for the recovery families);
a training set (seed 611161, indices 0..95) and a held-out set (seed 611162, indices 96..127) pass all three overlap
checks (instruction, seeded file, params) for every family the D2 pool may draw from; the params fix did not change a
single byte of the pre-D1 tasks. What still needs a live host: the GUI's own save path, the no-op child, the scope
probe, the reward audit, and the step-0 scan (docs/d2-pool.md).

    .venv-factory/bin/python -m pytest tests/test_library.py      (needs the generator pins: t6/requirements-factory.txt)
"""
import hashlib
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
pytest.importorskip("openpyxl")
pytest.importorskip("docx")
pytest.importorskip("pptx")

from test_d1 import score  # noqa: E402

from drums_tasks import build, split  # noqa: E402
from drums_tasks.core import canonical  # noqa: E402

NEW = ["medium_files_rules", "medium_calc_pipeline", "medium_writer_restructure", "multi_csv_report",
       "multi_letter_fill", "hidden_files_dotfiles", "hidden_calc_rows", "hidden_writer_readonly",
       "recover_files_interrupted", "recover_calc_column", "recover_writer_steps"]

# the pre-D1 families the audit keeps for the D2 pool (docs/task-library.md, the audit table)
KEPT = ["files_rename", "files_move_ext", "files_delete_pattern", "files_mkdir_tree", "files_zip", "files_count",
        "files_find_copy", "calc_aggregate", "calc_new_column", "calc_rename_sheet", "calc_delete_rows",
        "writer_replace", "writer_append", "writer_delete_para", "writer_heading",
        "impress_title", "impress_delete_slide", "impress_add_slide"]
D1 = ["d1_files_collect", "d1_calc_aggregate", "d1_writer_style"]
# every pre-D1 family on the apps Drums runs today got lossless params (kept, fix-later and dropped alike)
PARAMS_FIXED = KEPT + ["writer_bold", "files_concat", "calc_sort", "calc_export_csv", "writer_export_pdf", "multi_calc_to_text",
                       "multi_doc_to_sheet", "multi_sheet_to_doc"]
# held-out seeds other than 611162, from the C1 pool's seed sweep (runbooks/compare/20261117T0340Z/pool/POOL.md):
# with 611162 each has one instruction collision (the same wording over different data)
HELDOUT_SEED = {"calc_aggregate": 611169, "d1_calc_aggregate": 611163}

# sha256 of the canonical task JSON of variants 0..7 at seed 20261108, computed with drums-verifiers 0.2.2 (main at
# abd9755) before the params fix: params live in solutions.jsonl only, so the tasks must not change
TASKS_BEFORE = {
    "calc_aggregate": "31ed97fa01fc2615e08623f93660662072b268137d4ab834e8deada1d9417b38",
    "calc_delete_rows": "7f274746b76ae4c2aeb950a2c22424f946d90a2612c3e7d9f4329fb2bf040b0a",
    "calc_export_csv": "35d53d6a56daf039f1d900e870c50a58fb765f4000bb518c583c890964a6c313",
    "calc_new_column": "b5275241782f18a8126bc504da55d120e0f4ff49773f05d64698f1c4b2a05e9c",
    "calc_rename_sheet": "86fd49cdab1e5f534887c400c1f8b2ae130e0fb8aea028c05b04fe46144b75e4",
    "calc_sort": "47fc5595e4c42cf1938155d3ec7d47fe8398e75d4671d92835f5cb0b3c30f390",
    "files_concat": "c870d59d853f37f26c2ebcf7b6d0ccc74e7764a09bb28db49d273f36337c6550",
    "files_count": "5bfc40c7b9f17cbeb1e36a0d3b25ee6f6643563dcd4c2d4c7d89a330f71b69d8",
    "files_delete_pattern": "2b3241b8d6f6f459284ccb70d123e41952864e9424848d401507141bb47ff460",
    "files_find_copy": "1b2a6968f3a6ca638067358316e53d4c92b6c6346b914f42d19bac00e6f71aa8",
    "files_mkdir_tree": "fe5a6978f26ee97fd2585579bf1bccd2c9f6e98b7ff3f8175d3654d82a319f2f",
    "files_move_ext": "469cb3b9426a5bddc7b1b041d9796549575aa9c1bc2c9fcc867b1cb119ed7987",
    "files_rename": "281a1b804175eaac82f30dd7ddaf91f53f4e5e19fc3edfa7be74011e0dd7b9e5",
    "files_zip": "990e88280d62fb715eb881cfd55fe7dcfa40c256b4e1e6b0508639cfb452768d",
    "impress_add_slide": "8a38886187d04d64308300c9700893949732a35165f6c5216099e496cc9a027b",
    "impress_delete_slide": "0648d40ab4da830317ddb99afc04040960dd6498bc368a81bd6d722e320e2462",
    "impress_title": "d38f205acea7860aca95c9b355026d96712ebeffbc1216bf223607922d57e487",
    "multi_calc_to_text": "a358fcc0cfab979b05adc63e2a3f20d1bd2b751e5491e017cb7b6fbf9efd019d",
    "multi_doc_to_sheet": "3ba60108a2041bb784f564022184f9bf76a9ce48121bc252b09c4a99f9796318",
    "multi_sheet_to_doc": "e934bb5a917ea51744d44c7c53492427fbc939408e69a2b238c8212b72300031",
    "writer_append": "050168010762e2404577a70b842d01b6abf2e156b57672362198ee7b9276cf6d",
    "writer_bold": "b9e7d7311f422033e412fa2ed4d5573b0b9e6e5e673b9df1dd3966387bd2948d",
    "writer_delete_para": "e6a6a318f39f5ddfd556a5004871b1a10b5ddf87e1dec200a2949d46fe33357e",
    "writer_export_pdf": "d94c66e5b80f9224f0f777bba5a39da5de0974bbd0040b9b279f1c9530d01bbf",
    "writer_heading": "e65aa3778279366d2bb9a4291724d95d2f380eb7e1c6e304cdd3d0f3bc016378",
    "writer_replace": "a2ef597d2d4ad9c950ba436091b8c1e51ce3acb862e64d1fd8f8bf693c19b375",
}


def cases(fam, seed=20261117, count=32, start=0):
    tasks, sols = build.generate(seed, only={fam}, count=count, start=start)
    return list(zip(tasks, sols))


def test_every_family_is_registered_with_enough_variants():
    for fam in NEW:
        assert fam in build.ALL and build.ALL[fam][1] >= 96, fam
    assert len(NEW) >= 9


@pytest.mark.parametrize("fam", NEW)
def test_new_family_variants_differ(fam):
    cs = cases(fam, count=96)
    assert len({t["instruction"] for t, _ in cs}) == 96
    assert len({canonical(s["params"]) for _, s in cs}) == 96
    assert len({canonical(sorted(split.seeded_files(t).items())) for t, _ in cs}) == 96
    for t, s in cs:
        assert len(s["near_misses"]) >= 3, t["id"]
        assert len({m["name"] for m in s["near_misses"]}) == len(s["near_misses"])
        assert s["difficulty_hint"]["horizon"] in ("short", "medium", "long")


@pytest.mark.parametrize("fam", NEW)
def test_new_family_oracle_passes_near_misses_and_noop_fail(fam):
    for t, s in cases(fam):
        runs = [("oracle", s["oracle"], 1.0), ("no-op", [], 0.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]
        for name, actions, want in runs:
            r1, r2 = score(t, actions)
            assert r1["score"] == want, (t["id"], name, r1)
            assert r2["score"] == want, (t["id"], name, "alt", r2)


@pytest.mark.skipif(not shutil.which("soffice"), reason="soffice not installed")
@pytest.mark.parametrize("fam", ["medium_calc_pipeline", "medium_writer_restructure", "multi_csv_report",
                                 "multi_letter_fill", "hidden_calc_rows", "recover_calc_column", "recover_writer_steps"])
def test_office_oracles_through_libreoffice(fam):
    """The oracle's file written by LibreOffice itself (the writer a GUI save goes through) still separates."""
    for t, s in cases(fam, count=1):
        for name, actions, want in [("oracle", s["oracle"], 1.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]:
            r1, r2 = score(t, actions, soffice=True)
            assert r1["score"] == want, (t["id"], name, r1)
            assert r2["score"] == want, (t["id"], name, "alt", r2)


@pytest.mark.parametrize("fam", NEW + KEPT + D1)
def test_train_and_heldout_pass_all_three_overlap_checks(fam):
    """Zero collisions on instruction, seeded file and params between 96 training and 32 held-out variants."""
    train = cases(fam, seed=611161, count=96, start=0)
    held = cases(fam, seed=HELDOUT_SEED.get(fam, 611162), count=32, start=96)
    assert split.overlaps(train, held) == [], fam


@pytest.mark.parametrize("fam", KEPT)
def test_kept_family_oracle_passes_and_near_misses_fail(fam):
    """The office oracles go through the copy stand-in for LibreOffice (test_d1.OfficeCtx); every kept family now has at
    least three near misses."""
    for t, s in cases(fam, count=16):
        assert len(s["near_misses"]) >= 3, t["id"]
        for name, actions, want in [("oracle", s["oracle"], 1.0)] + [(m["name"], m["actions"], 0.0) for m in s["near_misses"]]:
            r1, r2 = score(t, actions)
            assert r1["score"] == want, (t["id"], name, r1)
            assert r2["score"] == want, (t["id"], name, "alt", r2)


@pytest.mark.parametrize("fam", PARAMS_FIXED)
def test_params_are_lossless(fam):
    """Every variant has its own params: they record each drawn parameter and a digest of the drawn content."""
    cs = cases(fam, count=96)
    by_params = {}
    for t, s in cs:
        key = canonical(s["params"])
        assert key not in by_params, (fam, t["id"], by_params.get(key))
        by_params[key] = t["id"]


@pytest.mark.parametrize("fam", sorted(TASKS_BEFORE))
def test_params_fix_leaves_the_tasks_unchanged(fam):
    got = hashlib.sha256(b"".join(canonical(t) + b"\n" for t, _ in cases(fam, seed=20261108, count=8))).hexdigest()
    assert got == TASKS_BEFORE[fam], fam


def test_split_start_moves_both_index_ranges(tmp_path):
    pool = split.split("hidden_files_dotfiles", 11, 12, str(tmp_path), 4, 2, start=200)
    assert pool["overlap"]["ok"]
    assert pool["train"]["ids"] == [f"t6-hidden-files-dotfiles-{i:04d}" for i in range(200, 204)]
    assert pool["heldout"]["ids"] == [f"t6-hidden-files-dotfiles-{i:04d}" for i in range(204, 206)]
    assert (pool["train"]["start"], pool["heldout"]["start"]) == (200, 204)
