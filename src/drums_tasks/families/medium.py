"""Medium-horizon families (about 15 to 50 GUI actions): several rules or steps stated once at the start, which the
agent must keep while it works through the whole task.

- medium_files_rules: sort an inbox folder by four rules with a precedence (file manager).
- medium_calc_pipeline: delete rows, add a computed column, sort by it, add a total row, rename the sheet (Calc).
- medium_writer_restructure: delete a section, rename a heading, then reorder the sections by heading (Writer).

Every drawn parameter is recorded in `params` (lossless), so the split's params check is a real duplicate check.
The checkers are existing kinds of drums_t6 (the verifier is not changed here)."""
from __future__ import annotations

from ..core import HOME, Spec, office_write, phrase, pick, sha256
from ..data import DATASETS, TOPICS, WORDS, sentence, table, xlsx_bytes
from .files import _content, _mkdir, _mv, _rm
from .lib import digest, docx_sections, tilde

BASES = [f"{HOME}/Documents", f"{HOME}/Desktop", f"{HOME}/Downloads"]
DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]


# ------------------------------------------------------------------------------------------------ files: rules

INBOXES = ["inbox", "incoming", "unsorted", "dropbox", "scans", "pile"]
DATA_DIRS = ["data", "tables", "sheets", "records"]
NOTE_DIRS = ["notes", "texts", "writing", "docs"]
DRAFT_DIRS = ["drafts", "pending", "unfinished", "wip_files"]
DRAFT_WORDS = ["draft", "wip", "tmp", "todo"]
DATA_EXT, NOTE_EXT = ("csv", "json"), ("md", "txt")


def _rule_dest(name, draft_word, dirs):
    """-> the folder key a file goes to (or None: deleted). The draft rule comes first."""
    if draft_word in name:
        return "drafts"
    ext = name.rsplit(".", 1)[1]
    if ext in DATA_EXT:
        return "data"
    if ext in NOTE_EXT:
        return "notes"
    return None                                     # .log: deleted


def files_rules(rng, idx):
    base = pick(rng, BASES)
    inbox = f"{base}/{pick(rng, INBOXES)}_{rng.randint(10, 99)}"
    dirs = {"data": pick(rng, DATA_DIRS), "notes": pick(rng, NOTE_DIRS), "drafts": pick(rng, DRAFT_DIRS)}
    draft = pick(rng, DRAFT_WORDS)
    for _ in range(200):
        n = rng.randint(9, 14)
        files, names = {}, set()
        while len(files) < n:
            ext = pick(rng, ["csv", "json", "md", "txt", "log"])
            w = pick(rng, WORDS)
            nm = f"{w}_{draft}_{rng.randint(1, 60)}.{ext}" if rng.random() < 0.25 else f"{w}_{rng.randint(1, 60)}.{ext}"
            if nm in names:
                continue
            names.add(nm)
            files[f"{inbox}/{nm}"] = _content(rng, ext, len(files))
        dest = {p: _rule_dest(p.rsplit("/", 1)[1], draft, dirs) for p in files}
        kinds = set(dest.values())
        # every rule has work to do, and at least one draft file would go elsewhere by its extension
        if {"data", "notes", "drafts", None} <= kinds and any(
                dest[p] == "drafts" and p.rsplit(".", 1)[1] != "log" for p in files):
            break
    else:
        raise RuntimeError("no usable inbox after 200 draws")

    def place(rule):
        out = {}
        for p, b in files.items():
            k = rule(p)
            if k is not None:
                out[f"{inbox}/{dirs[k]}/{p.rsplit('/', 1)[1]}"] = b
        return out
    want = place(lambda p: dest[p])
    listing = {inbox: sorted(dirs.values())}
    for k, v in dirs.items():
        listing[f"{inbox}/{v}"] = sorted(q.rsplit("/", 1)[1] for q in want if q.startswith(f"{inbox}/{v}/"))
    check = {"kind": "files_state", "must_exist": {p: sha256(b) for p, b in want.items()}, "must_absent": sorted(files),
             "dirs": [f"{inbox}/{v}" for v in dirs.values()], "listing": listing}
    mk = [_mkdir(*[f"{inbox}/{v}" for v in dirs.values()])]

    def moves(rule, keep=()):
        acts = []
        for p in sorted(files):
            if p in keep:
                continue
            k = rule(p)
            acts.append(_rm(p) if k is None else _mv(p, f"{inbox}/{dirs[k]}/"))
        return acts
    oracle = mk + moves(lambda p: dest[p])

    def by_ext(p):                                  # the extension rules only: the draft rule forgotten
        return _rule_dest(p.rsplit("/", 1)[1], "\0", dirs)
    left = pick(rng, [p for p in sorted(files) if dest[p] is not None])
    nm = [("draft_rule_ignored", mk + moves(by_ext)),
          ("logs_kept", mk + [a for a in moves(lambda p: dest[p]) if a["command"][0] != "rm"]),
          ("one_left_behind", mk + moves(lambda p: dest[p], keep=(left,))),
          ("copied_not_moved", mk + [dict(a, command=["cp", *a["command"][1:]]) if a["command"][0] == "mv" else a
                                     for a in moves(lambda p: dest[p])])]
    rules = (f"(1) any file whose name contains \"{draft}\" goes into {dirs['drafts']}, whatever its extension (this rule "
             f"comes first); (2) .csv and .json files go into {dirs['data']}; (3) .md and .txt files go into "
             f"{dirs['notes']}; (4) .log files are deleted")
    instr = phrase(rng, [
        "Sort the folder {i} by these rules, creating the three folders inside it: {r}. Move the files (do not copy "
        "them); when you are done {i} holds only the folders {d}, {n} and {f}.",
        "Tidy {i}. Make the sub-folders {d}, {n} and {f} in it and apply these rules to every file: {r}. Files are "
        "moved, not copied, and nothing but those three folders stays directly in {i}.",
        "Organise the files in {i} into three new sub-folders. The rules: {r}. Move each file; at the end only the "
        "folders {d}, {n} and {f} remain in {i}."],
        i=tilde(inbox), r=rules, d=dirs["data"], n=dirs["notes"], f=dirs["drafts"])
    return Spec(family="medium_files_rules", app="files", instruction=instr, files=files, check=check, oracle=oracle,
                near_misses=nm, open_cmd=["nautilus", inbox], allow=[f"{inbox}/*", f"{inbox}/*/*"], settle_s=3,
                difficulty={"horizon": "medium", "skills": ["constraint_retention", "rule_precedence"]},
                params={"inbox": inbox, "dirs": dirs, "draft": draft, "files": sorted(p.rsplit("/", 1)[1] for p in files),
                        "content": digest({p: sha256(b) for p, b in files.items()})})


# ------------------------------------------------------------------------------------------------ calc: pipeline

OPS = {"multiplied by": lambda a, b: a * b, "plus": lambda a, b: a + b, "minus": lambda a, b: a - b}
COLNAMES = ["Score", "Metric", "Result", "Combined", "Value", "Index"]
SHEETS = ["Report", "Final", "Clean", "Summary", "Review", "Totals"]


def _col(i):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def _pipeline(rows, filt, ai, bi, op, desc, total_rows=None):
    """Steps 1 to 4 on `rows`: keep rows passing filt, add the computed column, sort by it, a blank row, the Total row.
    total_rows: the rows the total is computed over (default: the kept rows)."""
    kept = [r for r in rows[1:] if filt(r)]
    body = sorted(([*r, OPS[op](r[ai], r[bi])] for r in kept), key=lambda r: r[-1], reverse=desc)
    tot_src = kept if total_rows is None else total_rows
    total = sum(OPS[op](r[ai], r[bi]) for r in tot_src)
    width = len(rows[0]) + 1
    return body, [None] * width, ["Total"] + [None] * (width - 2) + [total]


def calc_pipeline(rng, idx):
    for _ in range(300):
        ds = pick(rng, list(DATASETS))
        rows = table(rng, ds, rng.randint(10, 16))
        num = DATASETS[ds]["num"]
        fcol = pick(rng, num)
        fi = rows[0].index(fcol)
        vals = sorted({r[fi] for r in rows[1:]})
        if len(vals) < 6:
            continue
        thr = vals[rng.randint(2, len(vals) - 3)]
        less = rng.random() < 0.5
        a, b = rng.sample(num, 2)
        ai, bi = rows[0].index(a), rows[0].index(b)
        op = pick(rng, list(OPS))
        filt = (lambda r, t=thr, i=fi: r[i] >= t) if less else (lambda r, t=thr, i=fi: r[i] <= t)
        kept = [r for r in rows[1:] if filt(r)]
        dropped = [r for r in rows[1:] if not filt(r)]
        hv = [OPS[op](r[ai], r[bi]) for r in kept]
        # the computed values are distinct (one sort order), and the dropped rows change the total
        if len(set(hv)) == len(hv) and 3 <= len(kept) and dropped and sum(OPS[op](r[ai], r[bi]) for r in dropped) != 0:
            break
    else:
        raise RuntimeError("no usable pipeline after 300 tables")
    desc = rng.random() < 0.6
    name = f"{pick(rng, COLNAMES)}{rng.randint(1, 9)}"
    sheet = f"{pick(rng, SHEETS)}_{rng.randint(10, 99)}"
    d = pick(rng, DIRS)
    path = f"{d}/{DATASETS[ds]['file']}_{rng.randint(10, 99)}.xlsx"
    hcol = _col(len(rows[0]))
    header = [*rows[0], name]

    def table_for(filt2=filt, desc2=desc, total_rows=None):
        body, blank, tot = _pipeline(rows, filt2, ai, bi, op, desc2, total_rows)
        return [header, *body, blank, tot]
    want = table_for()

    def wb(rws, sh=sheet):
        return office_write(path, xlsx_bytes([(sh, rws)]), "xlsx")
    boundary = (lambda r: r[fi] > thr) if less else (lambda r: r[fi] < thr)
    nm = [("total_before_delete", wb(table_for(total_rows=rows[1:]))),
          ("boundary_rows_deleted", wb(table_for(filt2=boundary))),
          ("sorted_the_other_way", wb(table_for(desc2=not desc))),
          ("sheet_not_renamed", wb(want, "Sheet1")),
          ("no_rows_deleted", wb(table_for(filt2=lambda r: True)))]
    cmp_words = "less than" if less else "greater than"
    steps = (f"(1) delete every data row whose {fcol} is {cmp_words} {thr} (leave no empty rows); (2) in column {hcol} add "
             f"the header \"{name}\" and, for every remaining row, {a} {op} {b}; (3) sort the data rows by {name} "
             f"{'from largest to smallest' if desc else 'from smallest to largest'}, whole rows moving together; (4) leave "
             f"one empty row under the data and in the row after it put the word Total in column A and the sum of {name} "
             f"in column {hcol}; (5) rename the sheet to \"{sheet}\"")
    instr = phrase(rng, [
        "Work through {f} in this order: {s}. The header row stays on top. Save the file in its current format.",
        "The spreadsheet {f} is open. Do these steps one after another: {s}. Keep the header row and save the workbook "
        "as .xlsx.",
        "Edit {f}: {s}. Every later step works on the result of the earlier ones. Save it in the same format."],
        f=tilde(path), s=steps)
    check = {"kind": "all", "checks": [{"kind": "xlsx_sheetnames", "path": path, "names": [sheet]},
                                       {"kind": "xlsx_table", "path": path, "sheet": sheet, "rows": want, "tol": 0.01}]}
    return Spec(family="medium_calc_pipeline", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])}, check=check, oracle=wb(want), near_misses=nm,
                open_cmd=["libreoffice", "--calc", path], allow=[path], settle_s=8,
                difficulty={"horizon": "medium", "skills": ["constraint_retention", "dense_ui", "ordering"]},
                params={"dataset": ds, "path": path, "filter": {"column": fcol, "op": cmp_words, "value": thr},
                        "a": a, "b": b, "op": op, "name": name, "desc": desc, "sheet": sheet, "data": digest(rows)})


# ------------------------------------------------------------------------------------------------ writer: restructure

NEW_TITLES = ["Action items", "Background", "Costs", "Open risks", "Next quarter", "Lessons learned", "Zero-based review",
              "Approvals", "Milestones", "Wrap-up"]
DOC_NAMES = ["plan", "report", "handbook", "proposal", "notes", "review", "brief"]


def writer_restructure(rng, idx):
    for _ in range(300):
        k = rng.randint(4, 6)
        heads = rng.sample(TOPICS, k)
        seen, secs = set(), []
        for h in heads:
            body = []
            for _ in range(rng.randint(1, 3)):
                s = sentence(rng, len(seen))
                while s in seen:
                    s = sentence(rng, len(seen) + 50)
                seen.add(s)
                body.append(s)
            secs.append((h, body))
        di, ri = rng.sample(range(k), 2)
        new = f"{pick(rng, NEW_TITLES)} {rng.randint(2025, 2027)}"
        if new in heads:
            continue
        surv = [j for j in range(k) if j != di]                      # surviving sections, original order
        rest = [((new if j == ri else secs[j][0]), secs[j][1]) for j in surv]
        want = sorted(rest, key=lambda s: s[0].lower())
        # renamed, but ordered by the headings as they were before the rename
        by_old = [rest[m] for m in sorted(range(len(rest)), key=lambda m: secs[surv[m]][0].lower())]
        # the headings sorted, the paragraphs left where they were
        heads_only = [(h, b) for (h, _), (_, b) in zip(want, rest)]
        if want != rest and by_old != want and heads_only != want:
            break
    else:
        raise RuntimeError("no usable document after 300 draws")
    path = f"{pick(rng, DIRS)}/{pick(rng, DOC_NAMES)}_{rng.randint(10, 99)}.docx"

    def flat(ss):
        return [x for h, b in ss for x in [h, *b]]

    def doc(ss):
        return office_write(path, docx_sections(ss), "docx")
    not_renamed = sorted((secs[j] for j in surv), key=lambda s: s[0].lower())
    with_deleted = sorted(rest + [secs[di]], key=lambda s: s[0].lower())
    nm = [("sorted_by_old_heading", doc(by_old)),
          ("headings_moved_without_text", doc(heads_only)),
          ("section_not_deleted", doc(with_deleted)),
          ("heading_deleted_text_kept", doc(want + [("", secs[di][1])])),
          ("not_renamed", doc(not_renamed))]
    instr = phrase(rng, [
        "Restructure {f}: (1) delete the whole section \"{d}\" (its heading and every paragraph under it); (2) rename "
        "the heading \"{r}\" to \"{n}\"; (3) then reorder the sections so their headings are in alphabetical order, each "
        "heading keeping the paragraphs that follow it. Change no other text and save the document in the same format.",
        "In {f}, remove the section headed \"{d}\" with all its paragraphs, change the heading \"{r}\" to \"{n}\", and "
        "finally put the sections in alphabetical order of their headings (after the rename); every paragraph moves "
        "with its heading. Save as .docx in place.",
        "Edit {f} in three steps: drop the section \"{d}\" entirely, retitle \"{r}\" as \"{n}\", then sort the sections "
        "A to Z by heading, keeping each section's text under its heading. Nothing else changes; save it."],
        f=tilde(path), d=secs[di][0], r=secs[ri][0], n=new)
    return Spec(family="medium_writer_restructure", app="libreoffice_writer", instruction=instr,
                files={path: docx_sections(secs)},
                check={"kind": "docx_paragraphs", "path": path, "paragraphs": flat(want)},
                oracle=doc(want), near_misses=nm, open_cmd=["libreoffice", "--writer", path], allow=[path], settle_s=8,
                difficulty={"horizon": "medium", "skills": ["constraint_retention", "ordering", "dense_ui"]},
                params={"path": path, "sections": [h for h, _ in secs], "delete": secs[di][0], "rename": secs[ri][0],
                        "new": new, "text": digest(secs)})


FAMILIES = {"medium_files_rules": (files_rules, 96), "medium_calc_pipeline": (calc_pipeline, 96),
            "medium_writer_restructure": (writer_restructure, 96)}
