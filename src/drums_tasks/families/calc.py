"""LibreOffice Calc families: aggregate into a cell, sort, computed column, rename sheet, delete rows, export CSV."""
from __future__ import annotations

from ..core import HOME, Spec, lo_convert, office_write, phrase, pick
from ..data import DATASETS, csv_bytes, table, xlsx_bytes

DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]
AGG = {"sum": sum, "maximum": max, "minimum": min, "average": lambda v: sum(v) / len(v), "count": len}


def _col(i: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def _setup(rng, ds=None, n=None):
    ds = ds or pick(rng, list(DATASETS))
    n = n or rng.randint(8, 18)
    rows = table(rng, ds, n)
    d = pick(rng, DIRS)
    name = DATASETS[ds]["file"] + ("" if rng.random() < 0.5 else f"_{rng.randint(2024, 2026)}")
    path = f"{d}/{name}.xlsx"
    return ds, rows, path


def _cells(rows):
    return {f"{_col(j)}{i + 1}": v for i, r in enumerate(rows) for j, v in enumerate(r)}


def _write(path, rows, sheet="Sheet1"):
    return office_write(path, xlsx_bytes([(sheet, rows)]), "xlsx")


def _open(path):
    return ["libreoffice", "--calc", path]


def aggregate(rng, idx):
    ds, rows, path = _setup(rng)
    num = DATASETS[ds]["num"]
    colname = pick(rng, num)
    ci = rows[0].index(colname)
    agg = pick(rng, list(AGG))
    vals = [r[ci] for r in rows[1:]]
    want = AGG[agg](vals)
    tcol = _col(len(rows[0]) + 1)               # one empty column after the table
    target = f"{tcol}{rng.randint(1, 4)}"
    other = pick(rng, [a for a in AGG if AGG[a](vals) != want] or ["sum"])
    instr = phrase(rng, [
        "In {f}, put the {agg} of the {col} column (all data rows) into cell {t}, then save the file in its current format.",
        "Open {f} and write the {agg} of the values in column {col} in cell {t}. Keep everything else as it is and save.",
        "The spreadsheet {f} is open. Compute the {agg} of {col} over every row and enter it in {t}; save the workbook."],
        f=path.replace(HOME, "~"), agg=agg, col=colname, t=target)
    base = _cells(rows)

    def with_cell(ref, v):
        r2 = [list(r) + [None, None, None] for r in rows]
        rr, cc = int(ref[1:]) - 1, "ABCDEFGHIJ".index(ref[0])
        while len(r2) <= rr:
            r2.append([None] * len(r2[0]))
        r2[rr][cc] = v
        return r2
    oracle = _write(path, with_cell(target, want))
    nxt = _col(len(rows[0]) + 2) + target[1:]
    nm = [("wrong_aggregate", _write(path, with_cell(target, AGG[other](vals)))),
          ("wrong_cell", _write(path, with_cell(nxt, want))),
          ("missing_last_row", _write(path, with_cell(target, AGG[agg](vals[:-1]) if AGG[agg](vals[:-1]) != want else want + 1)))]
    return Spec(family="calc_aggregate", app="libreoffice_calc", instruction=instr, files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "xlsx_cells", "path": path, "sheet": None, "cells": {**base, target: want}, "tol": 0.01},
                oracle=oracle, near_misses=nm, open_cmd=_open(path), allow=[path], settle_s=8,
                params={"dataset": ds, "rows": len(rows) - 1, "agg": agg, "column": colname, "target": target})


def _unique_col(rows, cols):
    for c in cols:
        ci = rows[0].index(c)
        v = [r[ci] for r in rows[1:]]
        if len(set(v)) == len(v):
            return c
    return None


def sort_rows(rng, idx):
    for _ in range(50):
        ds, rows, path = _setup(rng)
        cols = list(rows[0])
        rng.shuffle(cols)
        key = _unique_col(rows, cols)
        if key:
            break
    ki = rows[0].index(key)
    desc = rng.random() < 0.5
    body = sorted(rows[1:], key=lambda r: r[ki], reverse=desc)
    want = [rows[0]] + body
    other = next(c for c in rows[0] if c != key)
    oi = rows[0].index(other)
    wrong_key = [rows[0]] + sorted(rows[1:], key=lambda r: (r[oi], r[ki]), reverse=desc)
    if wrong_key == want:
        wrong_key = [rows[0]] + body[::-1]
    only_col = [rows[0]] + [list(r) for r in rows[1:]]
    sorted_vals = [r[ki] for r in body]
    for i, r in enumerate(only_col[1:]):
        r[ki] = sorted_vals[i]
    if only_col == want:
        only_col = [rows[0]] + body[1:] + body[:1]
    instr = phrase(rng, [
        "Sort the data rows in {f} by {k} in {o} order (keep the header row on top) and save the file.",
        "In the open spreadsheet {f}, reorder the rows so {k} goes {o2}. Whole rows must move together. Save it.",
        "Sort {f} by the {k} column, {o}, keeping each row intact and the header first; then save."],
        f=path.replace(HOME, "~"), k=key, o="descending" if desc else "ascending",
        o2="from largest to smallest" if desc else "from smallest to largest")
    if isinstance(rows[1][ki], str):
        instr = instr.replace("from largest to smallest", "in reverse alphabetical order").replace(
            "from smallest to largest", "in alphabetical order")
    nm = [("reverse_order", _write(path, [rows[0]] + body[::-1])),
          ("wrong_key", _write(path, wrong_key)),
          ("column_only", _write(path, only_col))]
    return Spec(family="calc_sort", app="libreoffice_calc", instruction=instr, files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "xlsx_table", "path": path, "sheet": None, "rows": want, "tol": 1e-6},
                oracle=_write(path, want), near_misses=nm, open_cmd=_open(path), allow=[path], settle_s=8,
                params={"dataset": ds, "rows": len(rows) - 1, "key": key, "desc": desc})


OPS = {"product": ("multiplied by", lambda a, b: a * b), "sum": ("plus", lambda a, b: a + b),
       "difference": ("minus", lambda a, b: a - b)}


def new_column(rng, idx):
    ds, rows, path = _setup(rng)
    num = DATASETS[ds]["num"]
    a, b = rng.sample(num, 2)
    op = pick(rng, list(OPS))
    word, fn = OPS[op]
    name = pick(rng, ["Result", "Total", "Calc", "Combined", "Score", "Metric"]) + str(rng.randint(1, 9))
    ai, bi = rows[0].index(a), rows[0].index(b)
    col = _col(len(rows[0]))

    def table_with(fn2, header=name, shift=0):
        out = [list(rows[0]) + [header]]
        vals = [fn2(r[ai], r[bi]) for r in rows[1:]]
        if shift:
            vals = [None] + vals[:-1]
        for r, v in zip(rows[1:], vals):
            out.append(list(r) + [v])
        return out
    want = table_with(fn)
    other = pick(rng, [o for o in OPS if o != op])
    instr = phrase(rng, [
        "In {f}, add a new column {c} with the header \"{n}\" whose value in each row is {a} {w} {b}. Save the file.",
        "Add a column named \"{n}\" right after the last column of {f}: for every data row it holds {a} {w} {b}. Save.",
        "Open {f}. In column {c}, put the header {n} in row 1 and, for each data row, the value of {a} {w} {b}; then save."],
        f=path.replace(HOME, "~"), c=col, n=name, a=a, w=word, b=b)
    nm = [("wrong_operation", _write(path, table_with(OPS[other][1]))),
          ("shifted_one_row", _write(path, table_with(fn, shift=1))),
          ("missing_header", _write(path, table_with(fn, header=None)))]
    return Spec(family="calc_new_column", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "xlsx_table", "path": path, "sheet": None, "rows": want, "tol": 1e-6},
                oracle=_write(path, want), near_misses=nm, open_cmd=_open(path), allow=[path], settle_s=8,
                params={"dataset": ds, "a": a, "b": b, "op": op})


SHEETS = ["Data", "Summary", "Notes", "Q1", "Q2", "Archive", "Raw", "Input", "Budget", "Plan"]


def rename_sheet(rng, idx):
    ds, rows, path = _setup(rng)
    names = rng.sample(SHEETS, rng.randint(2, 4))
    k = rng.randrange(len(names))
    new = pick(rng, ["Report", "Final", "Overview", "Review", "Totals", "Clean"]) + "_" + str(rng.randint(10, 99))
    sheets = [(n, rows if i == 0 else [["note"], [n]]) for i, n in enumerate(names)]
    want = list(names)
    want[k] = new
    other = (k + 1) % len(names)
    wrong_sheet = list(names)
    wrong_sheet[other] = new
    typo = list(names)
    typo[k] = new.lower() if new.lower() != new else new + "x"

    def wb(ns):
        return office_write(path, xlsx_bytes([(n2, s[1]) for n2, s in zip(ns, sheets)]), "xlsx")
    instr = phrase(rng, [
        "Rename the sheet \"{o}\" in {f} to \"{n}\" and save the workbook.",
        "In {f}, the tab called {o} should be called {n}. Change it and save.",
        "Give the worksheet {o} of {f} the new name {n} (exact spelling), then save the file."],
        o=names[k], n=new, f=path.replace(HOME, "~"))
    return Spec(family="calc_rename_sheet", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes(sheets)},
                check={"kind": "xlsx_sheetnames", "path": path, "names": want},
                oracle=wb(want), near_misses=[("wrong_sheet", wb(wrong_sheet)), ("wrong_spelling", wb(typo))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"sheets": names, "k": k, "new": new})


def delete_rows(rng, idx):
    for _ in range(50):
        ds, rows, path = _setup(rng, n=rng.randint(10, 18))
        colname = pick(rng, DATASETS[ds]["num"])
        ci = rows[0].index(colname)
        vals = sorted({r[ci] for r in rows[1:]})
        if len(vals) >= 5:
            break
    thr = vals[len(vals) // 2]
    op = pick(rng, ["greater than", "less than"])
    keep = (lambda v: v <= thr) if op == "greater than" else (lambda v: v >= thr)
    keep_wrong = (lambda v: v < thr) if op == "greater than" else (lambda v: v > thr)
    want = [rows[0]] + [r for r in rows[1:] if keep(r[ci])]
    boundary = [rows[0]] + [r for r in rows[1:] if keep_wrong(r[ci])]
    complement = [rows[0]] + [r for r in rows[1:] if not keep(r[ci])]
    instr = phrase(rng, [
        "In {f}, delete every data row whose {c} is {op} {t}. Leave no empty rows behind, keep the header, and save.",
        "Remove all rows from {f} where {c} is {op} {t} (rows equal to {t} stay). Save the spreadsheet.",
        "Clean up {f}: rows with {c} {op} {t} must go; the other rows stay in their order with no gaps. Save it."],
        f=path.replace(HOME, "~"), c=colname, op=op, t=thr)
    nm = [("boundary_rows_deleted", _write(path, boundary)), ("complement_deleted", _write(path, complement)),
          ("emptied_not_deleted", _write(path, [rows[0]] + [r if keep(r[ci]) else [None] * len(r) for r in rows[1:]]
                                         + [[0] * len(rows[0])]))]
    return Spec(family="calc_delete_rows", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "xlsx_table", "path": path, "sheet": None, "rows": want, "tol": 1e-6},
                oracle=_write(path, want), near_misses=nm, open_cmd=_open(path), allow=[path], settle_s=8,
                params={"dataset": ds, "column": colname, "op": op, "threshold": thr})


def export_csv(rng, idx):
    ds, rows, path = _setup(rng)
    d = path.rsplit("/", 1)[0]
    stem = path.rsplit("/", 1)[1][:-5]
    out = f"{d}/{stem}.csv"
    instr = phrase(rng, [
        "Export the sheet in {f} as a comma-separated CSV file named {o} in the same folder (keep the header row).",
        "Save a CSV copy of {f} at {o}, with commas between fields and the header as the first line.",
        "Make a CSV version of the spreadsheet {f}: same folder, file name {n}, comma separated."],
        f=path.replace(HOME, "~"), o=out.replace(HOME, "~"), n=f"{stem}.csv")

    def csvw(rws, p=out):
        from ..core import write_file_steps
        return write_file_steps(p, csv_bytes(rws))
    other_dir = f"{HOME}/Desktop" if d.endswith("Documents") else f"{HOME}/Documents"
    nm = [("no_header", csvw(rows[1:])), ("missing_last_row", csvw(rows[:-1])),
          ("wrong_folder", csvw(rows, f"{other_dir}/{stem}.csv"))]
    return Spec(family="calc_export_csv", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "csv_table", "path": out, "rows": rows, "tol": 1e-6},
                oracle=[lo_convert(path, "csv", d)], near_misses=nm, open_cmd=_open(path),
                allow=[out, path], settle_s=8, params={"dataset": ds, "out": out})


FAMILIES = {"calc_aggregate": (aggregate, 90), "calc_sort": (sort_rows, 70), "calc_new_column": (new_column, 70),
            "calc_rename_sheet": (rename_sheet, 45), "calc_delete_rows": (delete_rows, 70),
            "calc_export_csv": (export_csv, 45)}
