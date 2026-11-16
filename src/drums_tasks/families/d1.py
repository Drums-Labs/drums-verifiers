"""The D1 study templates, one per app: collect files by pattern into a renamed set (files), a conditional aggregate
into a cell (Calc), bold every occurrence of a word and save under a new name (Writer).

Each family draws every parameter, file name, content and number from `rng`, so a seed gives a disjoint set of
variants from another seed (the split module checks that). The checkers are existing kinds of drums_t6: the verifier
is shipped byte-for-byte and is not changed here. That is why the Writer family applies bold only: `docx_bold` is the
one character-style check the verifier has (italic and highlight would need a new kind and its redundant twin)."""
from __future__ import annotations

import datetime
import io

from ..core import FIXED_ZIP_TIME, HOME, Spec, normalize_zip, office_write, phrase, pick, sha256
from ..data import CITIES, DATASETS, PRODUCTS, WORDS, docx_bytes, sentence, table, xlsx_bytes

BASES = [f"{HOME}/Documents", f"{HOME}/Desktop", f"{HOME}/Downloads"]
DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]


def _mkdir(*p):
    return {"type": "execute", "command": ["mkdir", "-p", *p], "shell": False}


def _cp(a, b):
    return {"type": "execute", "command": ["cp", a, b], "shell": False}


def _mv(a, b):
    return {"type": "execute", "command": ["mv", a, b], "shell": False}


def _tilde(p):
    return p.replace(HOME, "~")


# ------------------------------------------------------------------------------------------------ files: collect

TREE_ROOTS = ["project", "client", "archive", "inbox", "shared", "records"]
SUBS = ["data", "drafts", "notes", "reports", "old", "q1", "q2", "final", "raw", "logs"]
TARGETS = ["collected", "selection", "gathered", "bundle", "picked"]
TAGS = ["copy", "sel", "keep", "col", "x", "2026"]
EXTS = ["txt", "csv", "md", "log", "json"]


def _content(rng, ext, i):
    if ext == "csv":
        return f"id,city,value\n{i},{pick(rng, CITIES)},{rng.randint(1, 999)}\n".encode()
    if ext == "json":
        return ('{"id": %d, "product": "%s", "qty": %d}\n' % (i, pick(rng, PRODUCTS), rng.randint(1, 99))).encode()
    return (sentence(rng, i) + "\n").encode()


def _tree(rng):
    """A root folder with 2 or 3 sub-folders and 7 to 11 files spread over all of them; base names unique in the tree."""
    base = pick(rng, BASES)
    root = f"{base}/{pick(rng, TREE_ROOTS)}_{rng.randint(10, 99)}"
    subs = rng.sample(SUBS, rng.randint(2, 3))
    dirs = [root] + [f"{root}/{s}" for s in subs]
    exts = rng.sample(EXTS, 3)
    words = rng.sample(WORDS, 4)
    n = rng.randint(7, 11)
    files, names = {}, set()
    while len(files) < n:
        d = dirs[len(files)] if len(files) < len(dirs) else pick(rng, dirs)
        ext = pick(rng, exts)
        nm = f"{pick(rng, words)}_{rng.randint(1, 60)}.{ext}"
        if nm in names:
            continue
        names.add(nm)
        files[f"{d}/{nm}"] = _content(rng, ext, len(files))
    return base, root, subs, files


def _matches(pattern, path):
    name = path.rsplit("/", 1)[1]
    if pattern["kind"] == "ext":
        return name.endswith("." + pattern["value"])
    return name.startswith(pattern["value"] + "_")


def _renamed(scheme, path):
    parent, name = path.rsplit("/", 1)
    stem, ext = name.rsplit(".", 1)
    if scheme["kind"] == "folder_prefix":
        return f"{parent.rsplit('/', 1)[1]}_{name}"
    if scheme["kind"] == "tag_prefix":
        return f"{scheme['tag']}_{name}"
    return f"{stem}_{scheme['tag']}.{ext}"


def _pattern_words(pattern):
    if pattern["kind"] == "ext":
        return f"file ending in .{pattern['value']}", f"all the .{pattern['value']} files"
    return f"file whose name starts with \"{pattern['value']}_\"", f"all the files whose name starts with \"{pattern['value']}_\""


def _scheme_words(scheme, example):
    ex = f"{example.rsplit('/', 2)[1]}/{example.rsplit('/', 1)[1]} becomes {_renamed(scheme, example)}"
    if scheme["kind"] == "folder_prefix":
        return f"<folder>_<original name>, where <folder> is the name of the folder the file sits in ({ex})"
    if scheme["kind"] == "tag_prefix":
        return f"{scheme['tag']}_<original name> ({ex})"
    return f"<original name without extension>_{scheme['tag']}.<extension> ({ex})"


def files_collect(rng, idx):
    for _ in range(200):
        base, root, subs, files = _tree(rng)
        paths = sorted(files)
        n = len(paths)
        cands = []
        for e in sorted({p.rsplit(".", 1)[1] for p in paths}):
            k = sum(p.endswith("." + e) for p in paths)
            if 2 <= k <= n - 2:
                cands.append({"kind": "ext", "value": e})
        for w in sorted({p.rsplit("/", 1)[1].split("_")[0] for p in paths}):
            k = sum(p.rsplit("/", 1)[1].startswith(w + "_") for p in paths)
            if 2 <= k <= n - 2:
                cands.append({"kind": "prefix", "value": w})
        if cands:
            break
    else:
        raise RuntimeError("no usable pattern after 200 trees")
    pattern = pick(rng, cands)
    scheme = pick(rng, [{"kind": "folder_prefix"}, {"kind": "tag_prefix", "tag": pick(rng, TAGS)},
                        {"kind": "tag_suffix", "tag": pick(rng, TAGS)}])
    target = f"{base}/{pick(rng, TARGETS)}_{rng.randint(1, 99)}"
    hits = [p for p in paths if _matches(pattern, p)]
    rest = [p for p in paths if not _matches(pattern, p)]
    copies = {f"{target}/{_renamed(scheme, p)}": files[p] for p in hits}
    want = {**{p: sha256(b) for p, b in files.items()}, **{p: sha256(b) for p, b in copies.items()}}
    pat_one, pat_all = _pattern_words(pattern)
    instr = phrase(rng, [
        "Collect every {p1} from {r} and its sub-folders into a new folder {t}: copy them (the originals stay where they are) and name each copy {s}.",
        "Create the folder {t} and copy into it {pa} found anywhere under {r}, sub-folders included. Each copy is named {s}. Leave the originals in place and put nothing else in {t}.",
        "Gather {pa} under {r} (look in its sub-folders too) into {t} as copies named {s}. Keep the originals; no other file goes into {t}."],
        p1=pat_one, pa=pat_all, r=_tilde(root), t=_tilde(target), s=_scheme_words(scheme, hits[0]))
    oracle = [_mkdir(target)] + [_cp(p, f"{target}/{_renamed(scheme, p)}") for p in hits]
    extra = pick(rng, rest)
    nm = [("one_missed", [_mkdir(target)] + [_cp(p, f"{target}/{_renamed(scheme, p)}") for p in hits[:-1]]),
          ("non_matching_included", oracle + [_cp(extra, f"{target}/{_renamed(scheme, extra)}")]),
          ("not_renamed", [_mkdir(target)] + [_cp(p, f"{target}/{p.rsplit('/', 1)[1]}") for p in hits]),
          ("moved_not_copied", [_mkdir(target)] + [_mv(p, f"{target}/{_renamed(scheme, p)}") for p in hits])]
    return Spec(family="d1_files_collect", app="files", instruction=instr, files=files,
                check={"kind": "files_state", "must_exist": want, "dirs": [target],
                       "listing": {target: sorted(p.rsplit("/", 1)[1] for p in copies)}},
                oracle=oracle, near_misses=nm, open_cmd=["nautilus", root], allow=[target, f"{target}/*"], settle_s=3,
                params={"root": root, "subs": subs, "files": [p[len(root) + 1:] for p in paths], "pattern": pattern,
                        "scheme": scheme, "target": target, "matched": len(hits)})


# ------------------------------------------------------------------------------------------------ calc: conditional aggregate

AGG = {"sum": sum, "average": lambda v: sum(v) / len(v), "count": len, "maximum": max}
WHAT = {"sum": "the sum of {c}", "average": "the average of {c}", "maximum": "the largest {c} value"}


def _col(i):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def _cells(rows):
    return {f"{_col(j)}{i + 1}": v for i, r in enumerate(rows) for j, v in enumerate(r)}


def _with_cell(rows, ref, v):
    out = [list(r) + [None, None, None] for r in rows]
    rr, cc = int(ref[1:]) - 1, "ABCDEFGHIJ".index(ref[0])
    while len(out) <= rr:
        out.append([None] * len(out[0]))
    out[rr][cc] = v
    return out


def _conditions(rng, rows):
    """Every condition that keeps 2 to n-2 rows: equality on a text column, strict comparison on a number column."""
    n = len(rows) - 1
    out = []
    for j, name in enumerate(rows[0]):
        vals = [r[j] for r in rows[1:]]
        if isinstance(vals[0], str):
            for v in sorted(set(vals)):
                if 2 <= vals.count(v) <= n - 2:
                    out.append({"column": name, "op": "is", "value": v})
        else:
            for t in sorted(set(vals)):
                if 2 <= sum(x > t for x in vals) <= n - 2:
                    out.append({"column": name, "op": "greater than", "value": t})
                if 2 <= sum(x < t for x in vals) <= n - 2:
                    out.append({"column": name, "op": "less than", "value": t})
    return out


def _holds(cond, row, header):
    v = row[header.index(cond["column"])]
    if cond["op"] == "is":
        return v == cond["value"]
    return v > cond["value"] if cond["op"] == "greater than" else v < cond["value"]


def _apart(a, b):
    return abs(a - b) > 0.05


def calc_aggregate(rng, idx):
    for _ in range(200):
        ds = pick(rng, list(DATASETS))
        rows = table(rng, ds, rng.randint(10, 18))
        conds = _conditions(rng, rows)
        if not conds:
            continue
        cond = pick(rng, conds)
        agg = pick(rng, list(AGG))
        col = cond["column"] if agg == "count" else pick(rng, [c for c in DATASETS[ds]["num"] if c != cond["column"]])
        ci = rows[0].index(col)
        inside = [r[ci] for r in rows[1:] if _holds(cond, r, rows[0])]
        outside = [r[ci] for r in rows[1:] if not _holds(cond, r, rows[0])]
        every = [r[ci] for r in rows[1:]]
        want = AGG[agg](inside)
        other_agg = pick(rng, [a for a in AGG if a != agg and a != "count"])
        if agg == "count":
            other = AGG["sum"]([r[rows[0].index(DATASETS[ds]["num"][0])] for r in rows[1:] if _holds(cond, r, rows[0])])
        else:
            other = AGG[other_agg](inside)
        if all(_apart(want, x) for x in (AGG[agg](every), AGG[agg](outside), other)):
            break
    else:
        raise RuntimeError("no separable conditional aggregate after 200 tables")
    d = pick(rng, DIRS)
    name = DATASETS[ds]["file"] + ("" if rng.random() < 0.5 else f"_{rng.randint(2024, 2026)}")
    path = f"{d}/{name}.xlsx"
    target = f"{_col(len(rows[0]) + 1)}{rng.randint(1, 4)}"
    nxt = f"{_col(len(rows[0]) + 2)}{target[1:]}"
    cond_words = (f"{cond['column']} is \"{cond['value']}\"" if cond["op"] == "is"
                  else f"{cond['column']} is {cond['op']} {cond['value']}")
    what = (f"the number of data rows where {cond_words}" if agg == "count"
            else f"{WHAT[agg].format(c=col)} over the data rows where {cond_words}")
    instr = phrase(rng, [
        "In {f}, compute {w}, write the result in cell {t} and save the file in its current format.",
        "Open {f}. Work out {w} and put that one value in {t}. Change nothing else and save.",
        "The spreadsheet {f} is open. {W} goes into cell {t}. Save the workbook."],
        f=_tilde(path), w=what, W=what[0].upper() + what[1:], t=target)

    def write(ref, v):
        return office_write(path, xlsx_bytes([("Sheet1", _with_cell(rows, ref, v))]), "xlsx")
    nm = [("condition_ignored", write(target, AGG[agg](every))),
          ("wrong_aggregate", write(target, other)),
          ("wrong_cell", write(nxt, want)),
          ("complement_rows", write(target, AGG[agg](outside)))]
    return Spec(family="d1_calc_aggregate", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "xlsx_cells", "path": path, "sheet": None, "cells": {**_cells(rows), target: want}, "tol": 0.01},
                oracle=write(target, want), near_misses=nm, open_cmd=["libreoffice", "--calc", path], allow=[path], settle_s=8,
                params={"dataset": ds, "path": path, "rows": len(rows) - 1, "agg": agg, "column": col, "condition": cond,
                        "target": target, "value": want})


# ------------------------------------------------------------------------------------------------ writer: bold every occurrence

NAMES = ["memo", "minutes", "notes", "plan", "letter", "brief", "draft", "update", "agenda", "summary"]
SUFFIXES = ["bold", "marked", "styled", "v2", "final", "edited"]


def _docx_marked(paras, word, style, skip=0):
    """A .docx with every occurrence of `word` in one run carrying `style` (bold | italic); the last `skip` occurrences
    are left plain."""
    import docx
    total = sum(p.count(word) for p in paras)
    seen = 0
    d = docx.Document()
    for p in paras:
        para = d.add_paragraph()
        parts = p.split(word)
        for i, part in enumerate(parts):
            if part:
                para.add_run(part)
            if i < len(parts) - 1:
                seen += 1
                run = para.add_run(word)
                if seen <= total - skip:
                    setattr(run, style, True)
    t = datetime.datetime(*FIXED_ZIP_TIME)
    cp = d.core_properties
    cp.created = t
    cp.modified = t
    cp.author = "drums-t6"
    cp.last_modified_by = "drums-t6"
    cp.revision = 1
    b = io.BytesIO()
    d.save(b)
    return normalize_zip(b.getvalue())


def writer_style(rng, idx):
    for _ in range(200):
        paras, k = [], rng.randint(5, 8)
        while len(paras) < k:
            s = sentence(rng, len(paras))
            if s not in paras:
                paras.append(s)
        text = " ".join(paras)
        counts = {w: text.count(w) for w in WORDS if 2 <= text.count(w) <= 6}
        if counts:
            break
    else:
        raise RuntimeError("no repeated word after 200 documents")
    word = pick(rng, sorted(counts))
    n = counts[word]
    d = pick(rng, DIRS)
    stem = f"{pick(rng, NAMES)}_{rng.randint(1, 99):02d}"
    path = f"{d}/{stem}.docx"
    new = f"{d}/{stem}_{pick(rng, SUFFIXES)}.docx"
    other = pick(rng, [w for w in WORDS if w in text and w != word] or [word])
    instr = phrase(rng, [
        "In {f}, put every occurrence of the word \"{w}\" in bold (the text itself does not change) and save the document as {n} in the same folder, keeping the .docx format. Leave {f} itself as it is.",
        "Open {f}: make each \"{w}\" in the text bold, then use Save As to store the result under the name {n} (same folder, same format). The original file must stay unchanged.",
        "Every \"{w}\" in {f} should be bold. Apply that and save a copy named {n} next to it, in Word format; do not overwrite {f}."],
        f=_tilde(path), w=word, n=new.rsplit("/", 1)[1])
    original = docx_bytes(paras)

    def write(p, style="bold", skip=0):
        return office_write(p, _docx_marked(paras, word, style, skip), "docx")
    nm = [("one_missed", write(new, skip=1)),
          ("italic_instead", write(new, style="italic")),
          ("saved_over_original", write(path)),
          ("other_word", office_write(new, _docx_marked(paras, other, "bold") if other != word else docx_bytes(paras), "docx"))]
    return Spec(family="d1_writer_style", app="libreoffice_writer", instruction=instr, files={path: original},
                check={"kind": "all", "checks": [
                    {"kind": "docx_bold", "path": new, "paragraphs": paras, "bold": word * n},
                    {"kind": "files_state", "must_exist": {path: sha256(original)}}]},
                oracle=write(new), near_misses=nm, open_cmd=["libreoffice", "--writer", path], allow=[new], settle_s=8,
                params={"path": path, "new": new, "word": word, "occurrences": n, "paragraphs": len(paras)})


FAMILIES = {"d1_files_collect": (files_collect, 96), "d1_calc_aggregate": (calc_aggregate, 96), "d1_writer_style": (writer_style, 96)}
