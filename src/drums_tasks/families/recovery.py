"""Recovery families: the setup leaves the desktop in a broken or half-done state; the agent has to find what is wrong
and repair it. The fault is drawn per variant, so which rows, files or paragraphs are broken is not memorisable.

- recover_files_interrupted: a move of every file of one type into a sub-folder stopped halfway: some files moved,
  some not, one copy in the target truncated, a .part file and a "(copy)" duplicate left behind (file manager).
- recover_calc_column: a computed column has wrong numbers, empty cells and text placeholders in some rows (Calc).
- recover_writer_steps: a numbered procedure whose steps got out of order, with one step pasted twice (Writer).

A no-op scores 0 on every variant (the fault is real), and every drawn parameter, including the fault, is in
`params`. The checkers are existing kinds of drums_t6."""
from __future__ import annotations

from ..core import HOME, Spec, office_write, phrase, pick, sha256
from ..data import DATASETS, WORDS, docx_bytes, sentence, table, xlsx_bytes
from .files import _content, _cp, _mv, _rm
from .lib import digest, tilde

BASES = [f"{HOME}/Documents", f"{HOME}/Desktop", f"{HOME}/Downloads"]
DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]


# ------------------------------------------------------------------------------------------------ files: interrupted move

def files_interrupted(rng, idx):
    base = pick(rng, BASES)
    src = f"{base}/{pick(rng, ['export', 'inbox', 'batch', 'camera', 'reports', 'outbox'])}_{rng.randint(10, 99)}"
    ext = pick(rng, ["csv", "txt", "json", "md"])
    dst = f"{src}/{pick(rng, ['moved', 'sorted', 'done', 'by_type'])}_{ext}"
    others = [e for e in ["csv", "txt", "json", "md", "log"] if e != ext]
    names, full = set(), {}
    n_full, n_other = rng.randint(6, 8), rng.randint(2, 4)
    while len(full) < n_full:
        nm = f"{pick(rng, WORDS)}_{rng.randint(1, 60)}.{ext}"
        if nm not in names:
            names.add(nm)
            # long enough that a truncated copy is visibly shorter
            full[nm] = b"".join(_content(rng, ext, len(full) * 10 + i) for i in range(rng.randint(3, 6)))
    other = {}
    while len(other) < n_other:
        e = pick(rng, others)
        nm = f"{pick(rng, WORDS)}_{rng.randint(1, 60)}.{e}"
        if nm not in names:
            names.add(nm)
            other[nm] = _content(rng, e, 100 + len(other))
    order = sorted(full)
    rng.shuffle(order)
    # the fault: moved (only in dst), pending (only in src), truncated (src full + dst truncated), part (src full +
    # dst/<name>.part), duplicate (moved + "<stem> (copy).<ext>" in dst)
    moved, pending = order[:2], order[2:-3]
    truncated, part, dup = order[-3], order[-2], order[-1]
    moved = moved + [dup]
    files = {f"{src}/{n}": b for n, b in other.items()}
    for n in moved:
        files[f"{dst}/{n}"] = full[n]
    for n in pending + [truncated, part]:
        files[f"{src}/{n}"] = full[n]
    files[f"{dst}/{truncated}"] = full[truncated][:len(full[truncated]) // 2]
    files[f"{dst}/{part}.part"] = full[part][:len(full[part]) // 3]
    stem, e = dup.rsplit(".", 1)
    copy_name = f"{stem} (copy).{e}"
    files[f"{dst}/{copy_name}"] = full[dup]
    want = {**{f"{dst}/{n}": b for n, b in full.items()}, **{f"{src}/{n}": b for n, b in other.items()}}
    absent = sorted({f"{src}/{n}" for n in full} | {f"{dst}/{part}.part", f"{dst}/{copy_name}"})
    check = {"kind": "files_state", "must_exist": {p: sha256(b) for p, b in want.items()}, "must_absent": absent,
             "listing": {src: sorted([dst.rsplit("/", 1)[1], *other]), dst: sorted(full)}}
    fix_rest = [_mv(f"{src}/{n}", f"{dst}/{n}") for n in sorted(pending + [truncated, part])]
    clean = [_rm(f"{dst}/{part}.part", f"{dst}/{copy_name}")]
    oracle = fix_rest + clean
    nm = [("kept_truncated_copy", [_mv(f"{src}/{n}", f"{dst}/{n}") for n in sorted(pending + [part])]
           + [_rm(f"{src}/{truncated}")] + clean),
          ("part_file_left", fix_rest + [_rm(f"{dst}/{copy_name}")]),
          ("duplicate_left", fix_rest + [_rm(f"{dst}/{part}.part")]),
          ("copied_not_moved", [_cp(f"{src}/{n}", f"{dst}/{n}") for n in sorted(pending + [truncated, part])] + clean),
          ("other_types_moved_too", oracle + [_mv(f"{src}/{n}", f"{dst}/{n}") for n in sorted(other)])]
    instr = phrase(rng, [
        "A job that moves every .{e} file from {s} into {d} stopped halfway. Finish it and clean up: at the end every "
        ".{e} file is in {d} exactly once, under its original name and with its complete content; no .{e} file is left "
        "in {s}; the leftover .part file and any \"(copy)\" duplicate in {d} are gone; the other files in {s} stay put.",
        "Moving the .{e} files of {s} into {d} was interrupted. Repair the result: each .{e} file ends up once in {d} "
        "with its full original content and name, none stays in {s}, partial (.part) and duplicate \"(copy)\" files in {d} "
        "are deleted, and files of other types in {s} are not touched.",
        "Someone started moving all .{e} files from {s} to {d} and the move broke off. Complete it so that {d} holds every "
        ".{e} file once (original name, complete content, no .part or \"(copy)\" leftovers) and {s} keeps only its other "
        "files and the folder {dn}."],
        e=ext, s=tilde(src), d=tilde(dst), dn=dst.rsplit("/", 1)[1])
    return Spec(family="recover_files_interrupted", app="files", instruction=instr, files=files, check=check,
                oracle=oracle, near_misses=nm, open_cmd=["nautilus", src], allow=[f"{src}/*", f"{dst}/*"], settle_s=3,
                difficulty={"horizon": "medium", "skills": ["recovery", "self_verification"]},
                params={"src": src, "dst": dst, "ext": ext, "files": sorted(full), "other": sorted(other),
                        "fault": {"moved": sorted(moved), "pending": sorted(pending), "truncated": truncated, "part": part,
                                  "duplicate": dup},
                        "content": digest({n: sha256(b) for n, b in {**full, **other}.items()})})


# ------------------------------------------------------------------------------------------------ calc: broken column

OPS = {"multiplied by": lambda a, b: a * b, "plus": lambda a, b: a + b, "minus": lambda a, b: a - b}
COLNAMES = ["Total", "Value", "Amount", "Score", "Combined", "Result"]


def calc_column(rng, idx):
    for _ in range(200):
        ds = pick(rng, list(DATASETS))
        rows = table(rng, ds, rng.randint(9, 16))
        a, b = rng.sample(DATASETS[ds]["num"], 2)
        ai, bi = rows[0].index(a), rows[0].index(b)
        op = pick(rng, list(OPS))
        good = [OPS[op](r[ai], r[bi]) for r in rows[1:]]
        if len(set(good)) >= 4:
            break
    name = f"{pick(rng, COLNAMES)}_{rng.randint(1, 9)}"
    want = [[*rows[0], name]] + [[*r, v] for r, v in zip(rows[1:], good)]
    n = len(rows) - 1
    bad = rng.sample(range(n), rng.randint(3, 5))
    # one of each fault kind, the rest drawn: a wrong number, an empty cell, a text placeholder
    kinds = ["wrong", "empty", "text"] + [pick(rng, ["wrong", "empty", "text"]) for _ in bad[3:]]
    rng.shuffle(kinds)
    fault = {}
    broken = [list(r) for r in want]
    for i, k in zip(bad, kinds):
        if k == "wrong":
            v = good[i] + pick(rng, [-1, 1]) * rng.randint(1, 50)
        elif k == "empty":
            v = None
        else:
            v = pick(rng, ["TBD", "n/a", "#ERR", "?", "check"])
        broken[i + 1][-1] = v
        fault[str(i + 2)] = {"kind": k, "value": v}
    d = pick(rng, DIRS)
    path = f"{d}/{DATASETS[ds]['file']}_{rng.randint(100, 999)}.xlsx"
    col_letter = "ABCDEFGHIJ"[len(rows[0])]
    instr = phrase(rng, [
        "In {f}, the {c} column (column {l}) should hold {a} {o} {b} for every data row, but some of its cells are wrong, "
        "empty or hold text. Fix every bad cell, change nothing else, and save the file in its current format.",
        "The spreadsheet {f} was filled in carelessly: {c} must equal {a} {o} {b} in each row, and in some rows it does "
        "not. Find and correct those cells (leave the other columns alone) and save.",
        "Check the {c} column of {f}. Each value must be {a} {o} {b}; repair the rows where it is not (wrong numbers, "
        "blanks, placeholders) and save the workbook as .xlsx."],
        f=tilde(path), c=name, l=col_letter, a=a, o=op, b=b)

    def fixed(only):
        out = [list(r) for r in broken]
        for i in bad:
            if fault[str(i + 2)]["kind"] in only:
                out[i + 1][-1] = good[i]
        return out

    def wb(rws):
        return office_write(path, xlsx_bytes([("Sheet1", rws)]), "xlsx")
    inputs_changed = [list(r) for r in want]
    for i in bad:                                   # made consistent by changing an input instead of the result
        inputs_changed[i + 1][ai] = inputs_changed[i + 1][ai] + 1
        inputs_changed[i + 1][-1] = OPS[op](inputs_changed[i + 1][ai], inputs_changed[i + 1][bi])
    nm = [("blanks_and_text_only", wb(fixed({"empty", "text"}))),
          ("numbers_only", wb(fixed({"wrong"}))),
          ("bad_rows_deleted", wb([want[0]] + [r for i, r in enumerate(want[1:]) if i not in bad])),
          ("inputs_changed", wb(inputs_changed))]
    return Spec(family="recover_calc_column", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", broken)])},
                check={"kind": "xlsx_table", "path": path, "sheet": None, "rows": want, "tol": 0.01},
                oracle=wb(want), near_misses=nm, open_cmd=["libreoffice", "--calc", path], allow=[path], settle_s=8,
                difficulty={"horizon": "medium", "skills": ["recovery", "self_verification", "dense_ui"]},
                params={"dataset": ds, "path": path, "a": a, "b": b, "op": op, "name": name, "fault": fault,
                        "data": digest(rows)})


# ------------------------------------------------------------------------------------------------ writer: steps

TITLES = ["How to file an expense claim", "Onboarding checklist", "Release procedure", "Office move steps",
          "Quarterly close", "Vendor setup", "Backup routine", "Event planning", "Laptop return", "Audit preparation"]


def writer_steps(rng, idx):
    n = rng.randint(6, 12)
    seen, steps = set(), []
    while len(steps) < n:
        s = sentence(rng, len(steps))
        if s not in seen:
            seen.add(s)
            steps.append(f"Step {len(steps) + 1}: {s}")
    title = f"{pick(rng, TITLES)} ({pick(rng, ['v', 'rev ', 'draft '])}{rng.randint(1, 9)})"
    # the fault: two or three swaps, then one step pasted a second time somewhere after its first place
    order = list(range(n))
    swaps = []
    for _ in range(rng.randint(2, 3)):
        i, j = sorted(rng.sample(range(n), 2))
        order[i], order[j] = order[j], order[i]
        swaps.append([i, j])
    if order == list(range(n)):
        order[0], order[1] = order[1], order[0]
        swaps.append([0, 1])
    dup = rng.randrange(n)
    at = rng.randint(order.index(dup) + 1, n)
    broken_idx = order[:at] + [dup] + order[at:]
    broken = [title] + [steps[i] for i in broken_idx]
    want = [title] + steps
    path = f"{pick(rng, DIRS)}/{pick(rng, ['procedure', 'howto', 'steps', 'checklist', 'guide'])}_{rng.randint(10, 99)}.docx"

    def w(paras):
        return office_write(path, docx_bytes(paras), "docx")
    lexical = [title] + sorted(steps)               # "Step 10" before "Step 2"
    one_swap_left = list(want)
    a, b = rng.sample(range(n), 2)
    one_swap_left[a + 1], one_swap_left[b + 1] = one_swap_left[b + 1], one_swap_left[a + 1]
    nm = [("duplicate_left", w([title] + sorted((steps[i] for i in broken_idx), key=lambda s: int(s.split(":")[0][5:])))),
          ("order_not_fixed", w([title] + [steps[i] for i in order])),
          ("one_swap_left", w(one_swap_left)),
          ("both_copies_deleted", w([title] + [s for i, s in enumerate(steps) if i != dup]))]
    if n >= 10:
        nm.append(("sorted_as_text", w(lexical)))
    instr = phrase(rng, [
        "The steps in {f} got out of order and one step was pasted twice. Put the steps back in order from 1 to {n}, "
        "each step exactly once, keep the title line first, change no wording, and save the document in the same format.",
        "Repair {f}: its numbered steps (Step 1 to Step {n}) are shuffled and one of them appears twice. Reorder them by "
        "number, remove the extra copy, leave the title and the text of each step unchanged, and save.",
        "Someone messed up {f}: fix it so the title comes first and then Step 1, Step 2, ... Step {n}, in that order, "
        "with no step repeated. Do not reword anything. Save it as .docx."],
        f=tilde(path), n=n)
    return Spec(family="recover_writer_steps", app="libreoffice_writer", instruction=instr,
                files={path: docx_bytes(broken)},
                check={"kind": "docx_paragraphs", "path": path, "paragraphs": want},
                oracle=w(want), near_misses=nm, open_cmd=["libreoffice", "--writer", path], allow=[path], settle_s=8,
                difficulty={"horizon": "medium" if n >= 9 else "short", "skills": ["recovery", "ordering"]},
                params={"path": path, "title": title, "n": n, "swaps": swaps, "duplicate": dup, "at": at,
                        "text": digest(steps)})


FAMILIES = {"recover_files_interrupted": (files_interrupted, 96), "recover_calc_column": (calc_column, 96),
            "recover_writer_steps": (writer_steps, 96)}
