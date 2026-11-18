"""Hidden-state families: the desktop is in a state or mode the screen does not show at first glance, and the task
can only be done right by noticing it.

- hidden_files_dotfiles: some of the files to move are dotfiles, which the file manager hides by default; the
  instruction states how many there are, so the agent can check its own count (file manager).
- hidden_calc_rows: some data rows of the sheet are hidden; the aggregate is over every row (Calc).
- hidden_writer_readonly: the document opens in read-only (viewer) mode; it must be switched to editing before the
  change can be saved in place (Writer).

Every drawn parameter is in `params`. The checkers are existing kinds of drums_t6."""
from __future__ import annotations

from ..core import HOME, Spec, office_write, phrase, pick, sha256
from ..data import DATASETS, WORDS, docx_bytes, sentence, table, xlsx_bytes
from .files import _content, _cp, _mkdir, _mv
from .lib import digest, tilde, xlsx_hidden

BASES = [f"{HOME}/Documents", f"{HOME}/Desktop", f"{HOME}/Downloads"]
DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]


# ------------------------------------------------------------------------------------------------ files: dotfiles

def files_dotfiles(rng, idx):
    base = pick(rng, BASES)
    d = f"{base}/{pick(rng, ['project', 'share', 'export', 'sync', 'backup', 'work'])}_{rng.randint(10, 99)}"
    ext = pick(rng, ["txt", "csv", "md", "log", "json"])
    others = [e for e in ["txt", "csv", "md", "log", "json"] if e != ext]
    files, names = {}, set()
    n_vis, n_hid, n_other = rng.randint(2, 5), rng.randint(1, 3), rng.randint(2, 5)
    plan = [(ext, False)] * n_vis + [(ext, True)] * n_hid + [(pick(rng, others), False) for _ in range(n_other)] + \
        [(pick(rng, others), True)]                 # one hidden file of another type, which stays
    for e, hid in plan:
        while True:
            nm = f"{'.' if hid else ''}{pick(rng, WORDS)}_{rng.randint(1, 60)}.{e}"
            if nm not in names and nm.lstrip(".") not in {x.lstrip(".") for x in names}:
                break
        names.add(nm)
        files[f"{d}/{nm}"] = _content(rng, e, len(files))
    sub = f"{d}/{pick(rng, ['all', 'every', 'moved', 'sorted'])}_{ext}"
    movers = [p for p in sorted(files) if p.endswith("." + ext)]
    hidden_movers = [p for p in movers if p.rsplit("/", 1)[1].startswith(".")]
    stay = [p for p in sorted(files) if p not in movers]
    want = {**{f"{sub}/{p.rsplit('/', 1)[1]}": files[p] for p in movers}, **{p: files[p] for p in stay}}
    check = {"kind": "files_state", "must_exist": {p: sha256(b) for p, b in want.items()}, "must_absent": movers,
             "dirs": [sub], "listing": {d: sorted([sub.rsplit("/", 1)[1]] + [p.rsplit("/", 1)[1] for p in stay]),
                                        sub: sorted(p.rsplit("/", 1)[1] for p in movers)}}
    mk = [_mkdir(sub)]
    hid_other = next(p for p in stay if p.rsplit("/", 1)[1].startswith("."))
    nm = [("visible_only", mk + [_mv(p, sub + "/") for p in movers if p not in hidden_movers]),
          ("dot_dropped", mk + [_mv(p, f"{sub}/{p.rsplit('/', 1)[1].lstrip('.')}") for p in movers]),
          ("copied_not_moved", mk + [_cp(p, sub + "/") for p in movers]),
          ("other_hidden_moved_too", mk + [_mv(p, sub + "/") for p in movers + [hid_other]])]
    k = len(movers)
    instr = phrase(rng, [
        "The folder {d} contains exactly {k} .{e} files. Create the folder {s} and move all {k} of them into it; every "
        "other file stays where it is.",
        "Move every .{e} file in {d} ({k} in all, some of them may be hidden) into a new folder {s}. Leave the other files "
        "alone.",
        "There are {k} files ending in .{e} directly inside {d}. Put all of them, with their names unchanged, into the new "
        "sub-folder {s}; nothing else moves."],
        d=tilde(d), k=k, e=ext, s=sub.rsplit("/", 1)[1])
    return Spec(family="hidden_files_dotfiles", app="files", instruction=instr, files=files, check=check,
                oracle=mk + [_mv(p, sub + "/") for p in movers], near_misses=nm, open_cmd=["nautilus", d],
                allow=[f"{d}/*", f"{d}/.*", f"{sub}/*", f"{sub}/.*"], settle_s=3,
                difficulty={"horizon": "short", "skills": ["hidden_state", "self_verification"]},
                params={"dir": d, "ext": ext, "sub": sub, "files": sorted(names), "content": digest(
                    {p: sha256(b) for p, b in files.items()})})


# ------------------------------------------------------------------------------------------------ calc: hidden rows

AGG = {"sum": sum, "maximum": max, "minimum": min, "average": lambda v: sum(v) / len(v), "count": len}


def _col(i):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def _with_cell(rows, ref, v):
    out = [list(r) + [None, None, None] for r in rows]
    rr, cc = int(ref[1:]) - 1, "ABCDEFGHIJ".index(ref[0])
    while len(out) <= rr:
        out.append([None] * len(out[0]))
    out[rr][cc] = v
    return out


def calc_hidden_rows(rng, idx):
    for _ in range(300):
        ds = pick(rng, list(DATASETS))
        rows = table(rng, ds, rng.randint(10, 18))
        col = pick(rng, DATASETS[ds]["num"])
        ci = rows[0].index(col)
        agg = pick(rng, list(AGG))
        vals = [r[ci] for r in rows[1:]]
        n = len(vals)
        # the hidden rows: 2 to 4 data rows; for maximum/minimum the extreme row is one of them
        k = rng.randint(2, 4)
        hid = set(rng.sample(range(n), k))
        if agg == "maximum":
            hid.add(vals.index(max(vals)))
        if agg == "minimum":
            hid.add(vals.index(min(vals)))
        visible = [v for i, v in enumerate(vals) if i not in hid]
        want, vis = AGG[agg](vals), AGG[agg](visible)
        if abs(want - vis) > 0.05 and len(hid) < n - 2:
            break
    else:
        raise RuntimeError("no usable hidden-row table after 300 draws")
    hidden = sorted(i + 2 for i in hid)            # 1-based sheet rows (row 1 is the header)
    d = pick(rng, DIRS)
    path = f"{d}/{DATASETS[ds]['file']}_{rng.randint(10, 99)}.xlsx"
    target = f"{_col(len(rows[0]) + 1)}{rng.randint(1, 4)}"
    nxt = f"{_col(len(rows[0]) + 2)}{target[1:]}"
    what = "the number of data rows" if agg == "count" else f"the {agg} of {col}"
    instr = phrase(rng, [
        "In {f}, put {w} over all data rows of the sheet into cell {t}, then save the file in its current format.",
        "Open {f} and write {w}, counting every data row in the sheet, in cell {t}. Change nothing else and save.",
        "The spreadsheet {f} is open. Compute {w} over every data row (including any that are hidden) and enter it "
        "in {t}; save the workbook."],
        f=tilde(path), w=what, t=target)

    def write(rws, ref, v, hide=hidden):
        return office_write(path, xlsx_hidden(_with_cell(rws, ref, v), hide), "xlsx")
    shown = [rows[0]] + [r for i, r in enumerate(rows[1:]) if i not in hid]
    other = pick(rng, [a for a in AGG if abs(AGG[a](vals) - want) > 0.05] or ["sum"])
    nm = [("visible_rows_only", write(rows, target, vis)),
          ("hidden_rows_deleted", office_write(path, xlsx_bytes([("Sheet1", _with_cell(shown, target, vis))]), "xlsx")),
          ("wrong_cell", write(rows, nxt, want)),
          ("wrong_aggregate", write(rows, target, AGG[other](vals)))]
    cells = {f"{_col(j)}{i + 1}": v for i, r in enumerate(rows) for j, v in enumerate(r)}
    return Spec(family="hidden_calc_rows", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_hidden(rows, hidden)},
                check={"kind": "xlsx_cells", "path": path, "sheet": None, "cells": {**cells, target: want}, "tol": 0.01},
                oracle=write(rows, target, want), near_misses=nm, open_cmd=["libreoffice", "--calc", path],
                allow=[path], settle_s=8,
                difficulty={"horizon": "short", "skills": ["hidden_state", "dense_ui"]},
                params={"dataset": ds, "path": path, "agg": agg, "column": col, "hidden": hidden, "target": target,
                        "value": want, "data": digest(rows)})


# ------------------------------------------------------------------------------------------------ writer: read-only

NAMES = ["memo", "minutes", "notes", "plan", "letter", "brief", "draft", "update", "agenda", "summary"]
NEW_WORDS = ["plan", "draft", "audit", "roadmap", "ticket", "order", "memo", "quote", "survey", "ledger"]


def writer_readonly(rng, idx):
    for _ in range(200):
        paras, k = [], rng.randint(4, 7)
        while len(paras) < k:
            s = sentence(rng, len(paras))
            if s not in paras:
                paras.append(s)
        text = " ".join(paras)
        counts = {w: text.count(w) for w in WORDS if 2 <= text.count(w) <= 6}
        fresh = [w for w in NEW_WORDS if w not in text]
        if counts and fresh:
            break
    else:
        raise RuntimeError("no usable document after 200 draws")
    old = pick(rng, sorted(counts))
    new = pick(rng, fresh)
    d = pick(rng, DIRS)
    stem = f"{pick(rng, NAMES)}_{rng.randint(10, 99)}"
    path = f"{d}/{stem}.docx"
    want = [p.replace(old, new) for p in paras]
    first, done = [], False
    for p in paras:
        if not done and old in p:
            first.append(p.replace(old, new, 1))
            done = True
        else:
            first.append(p)
    instr = phrase(rng, [
        "In {f}, replace every \"{o}\" with \"{n}\" and save the document in place (same name, same format).",
        "The document {f} is open. Change each occurrence of the word \"{o}\" to \"{n}\" and save it over the original "
        "file.",
        "Edit {f}: every \"{o}\" becomes \"{n}\". The result must be saved to {f} itself as .docx."],
        f=tilde(path), o=old, n=new)
    copy = f"{d}/{stem}_edited.docx"
    nm = [("saved_as_copy", office_write(copy, docx_bytes(want), "docx")),
          ("first_only", office_write(path, docx_bytes(first), "docx")),
          ("replaced_other_word", office_write(path, docx_bytes([p.replace(old, old.upper()) for p in paras]), "docx"))]
    return Spec(family="hidden_writer_readonly", app="libreoffice_writer", instruction=instr,
                files={path: docx_bytes(paras)},
                check={"kind": "docx_paragraphs", "path": path, "paragraphs": want},
                oracle=office_write(path, docx_bytes(want), "docx"), near_misses=nm,
                # viewer mode: the document opens read-only; editing needs the infobar's Edit Document (or Edit > Edit Mode)
                open_cmd=["libreoffice", "--writer", "--view", path], allow=[path], settle_s=8,
                difficulty={"horizon": "short", "skills": ["hidden_state", "dialog"]},
                params={"path": path, "old": old, "new": new, "occurrences": counts[old], "text": digest(paras)})


FAMILIES = {"hidden_files_dotfiles": (files_dotfiles, 96), "hidden_calc_rows": (calc_hidden_rows, 96),
            "hidden_writer_readonly": (writer_readonly, 96)}
