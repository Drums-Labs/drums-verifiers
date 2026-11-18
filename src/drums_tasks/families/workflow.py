"""Multi-app workflows on the apps Drums runs today (file manager, LibreOffice Calc and Writer): data comes from one
file, is worked on in Calc and ends up in a Writer document, and the checker reads every output.

- multi_csv_report: a CSV export goes into a workbook, and a Writer report lists the per-category totals, largest
  first, and the grand total.
- multi_letter_fill: a Writer letter template is filled with one customer's row from a Calc sheet and saved under a new
  name; the template stays unchanged.

Every drawn parameter is in `params`. The checkers are existing kinds of drums_t6 (`all` of several sub-checks)."""
from __future__ import annotations

from ..core import HOME, Spec, office_write, phrase, pick, sha256
from ..data import CITIES, FIRST, LAST, csv_bytes, docx_bytes, table, xlsx_bytes
from .lib import digest, tilde

DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]


def _totals(rows):
    out = {}
    for r in rows[1:]:
        out[r[1]] = out.get(r[1], 0) + r[2]
    return out


def csv_report(rng, idx):
    for _ in range(200):
        rows = table(rng, "expenses", rng.randint(8, 14))
        tot = _totals(rows)
        less = _totals(rows[:-1])
        if len(tot) >= 3 and len(set(tot.values())) == len(tot) and \
                sorted(tot, key=tot.get, reverse=True) != sorted(tot):
            break
    else:
        raise RuntimeError("no usable expense table after 200 draws")
    month = f"{pick(rng, MONTHS)} {rng.randint(2024, 2026)}"
    d = pick(rng, DIRS)
    tag = f"{month.split()[0].lower()[:3]}{month.split()[1][2:]}_{rng.randint(100, 999)}"
    src = f"{d}/expenses_{tag}.csv"
    xl = f"{d}/expenses_{tag}.xlsx"
    rep = f"{d}/report_{tag}.docx"

    def report(t, order=None, grand=True):
        cats = order or sorted(t, key=t.get, reverse=True)
        return [f"Expense report {month}"] + [f"{c}: {t[c]}" for c in cats] + (
            [f"Grand total: {sum(t.values())}"] if grand else [])
    want = report(tot)
    check = {"kind": "all", "checks": [
        {"kind": "xlsx_table", "path": xl, "sheet": None, "rows": rows, "tol": 0.01},
        {"kind": "docx_paragraphs", "path": rep, "paragraphs": want},
        {"kind": "files_state", "must_exist": {src: sha256(csv_bytes(rows))}}]}

    def outs(xl_rows, paras):
        acts = office_write(xl, xlsx_bytes([("Sheet1", xl_rows)]), "xlsx") if xl_rows is not None else []
        return acts + office_write(rep, docx_bytes(paras), "docx")
    nm = [("alphabetical_order", outs(rows, report(tot, order=sorted(tot)))),
          ("report_only", outs(None, want)),
          ("grand_total_missing", outs(rows, report(tot, grand=False))),
          ("last_row_dropped", outs(rows[:-1], report(less)))]
    instr = phrase(rng, [
        "{s} is this month's expense export. (1) Make a workbook {x} with the same rows and columns (header first). "
        "(2) Write a Writer document {r} whose paragraphs are: the line \"Expense report {m}\", then one line per "
        "Category in the form \"Category: total Amount\" from the largest total to the smallest, then \"Grand total: N\" "
        "with the sum of all amounts. Keep {s} as it is.",
        "Turn the CSV {s} into a spreadsheet {x} (same table), then use it to write the report {r}: first paragraph "
        "\"Expense report {m}\", then for each Category a paragraph \"<Category>: <sum of Amount>\", biggest sum first, "
        "and last \"Grand total: <sum of every Amount>\". Do not change the CSV.",
        "Two outputs from {s}: the workbook {x} holding the same rows, and the document {r} that reads \"Expense report "
        "{m}\", then \"Category: total\" for every category (largest total first), then \"Grand total: N\". The CSV "
        "stays unchanged."],
        s=tilde(src), x=tilde(xl), r=tilde(rep), m=month)
    return Spec(family="multi_csv_report", app="libreoffice_calc", instruction=instr, files={src: csv_bytes(rows)},
                check=check, oracle=outs(rows, want), near_misses=nm, open_cmd=["libreoffice", "--calc", src],
                allow=[xl, rep], settle_s=8, multi_app=True,
                difficulty={"horizon": "medium", "skills": ["multi_app", "aggregation", "constraint_retention"]},
                params={"src": src, "xlsx": xl, "report": rep, "month": month, "data": digest(rows)})


TEMPLATES = [
    ["Dear [NAME],", "Our records show an open balance of [BALANCE] on your [CITY] account.",
     "Please settle it by day [DUE] of this month.", "Kind regards,"],
    ["Hello [NAME],", "This is a reminder that [BALANCE] is still due for the [CITY] office.",
     "The payment date is day [DUE].", "Thank you,"],
    ["To [NAME],", "Your account in [CITY] has a balance of [BALANCE].",
     "We expect payment no later than day [DUE].", "Best wishes,"],
]


def letter_fill(rng, idx):
    names, n = set(), rng.randint(6, 12)
    while len(names) < n:
        names.add(f"{pick(rng, FIRST)} {pick(rng, LAST)}")
    names = sorted(names)
    rng.shuffle(names)
    rows = [["Name", "City", "Balance", "Due"]] + [[n, pick(rng, CITIES), rng.randint(40, 9000), rng.randint(1, 28)]
                                                  for n in names]
    k = rng.randrange(1, len(rows))
    who = rows[k]
    nb = rows[k + 1] if k + 1 < len(rows) else rows[k - 1]
    tpl = [*pick(rng, TEMPLATES), f"{pick(rng, FIRST)} {pick(rng, LAST)}, Accounts"]
    d = pick(rng, DIRS)
    xl = f"{d}/customers_{rng.randint(10, 99)}.xlsx"
    template = f"{d}/letter_template_{rng.randint(10, 99)}.docx"
    out = f"{d}/letter_{who[0].lower().replace(' ', '_')}.docx"

    def fill(r, keys=("NAME", "CITY", "BALANCE", "DUE")):
        vals = {"NAME": r[0], "CITY": r[1], "BALANCE": str(r[2]), "DUE": str(r[3])}
        out_p = []
        for p in tpl:
            for key in keys:
                p = p.replace(f"[{key}]", vals[key])
            out_p.append(p)
        return out_p
    want = fill(who)
    check = {"kind": "all", "checks": [
        {"kind": "docx_paragraphs", "path": out, "paragraphs": want},
        {"kind": "files_state", "must_exist": {template: sha256(docx_bytes(tpl))}}]}
    left = pick(rng, ["CITY", "BALANCE", "DUE"])
    nm = [("neighbour_row", office_write(out, docx_bytes(fill([who[0], *nb[1:]])), "docx")),
          ("placeholder_left", office_write(out, docx_bytes(fill(who, tuple(x for x in ("NAME", "CITY", "BALANCE", "DUE")
                                                                             if x != left))), "docx")),
          ("saved_over_template", office_write(template, docx_bytes(want), "docx")),
          ("both_saved", office_write(template, docx_bytes(want), "docx") + office_write(out, docx_bytes(want), "docx")),
          ("balance_formatted", office_write(out, docx_bytes([p.replace(str(who[2]), f"{who[2]:,}.00") for p in want]),
                                             "docx"))]
    if f"{who[2]:,}.00" == str(who[2]):
        nm = nm[:-1]
    instr = phrase(rng, [
        "Fill in the letter template {t} for the customer {n}: replace [NAME], [CITY], [BALANCE] and [DUE] with that "
        "customer's Name, City, Balance and Due values from {x} (numbers exactly as in the sheet), and save the result as "
        "{o}. The template itself must not change.",
        "Using the row for {n} in {x}, complete the placeholders in {t} ([NAME], [CITY], [BALANCE], [DUE]; plain "
        "numbers) and save it under the new name {o}, in Word format. Leave the template as it is.",
        "Write {n}'s letter: open {t}, put in their name, city, balance and due day from {x} where the bracketed "
        "placeholders are (numbers as written in the sheet), then Save As {o}. Do not overwrite the template."],
        t=tilde(template), n=who[0], x=tilde(xl), o=tilde(out))
    return Spec(family="multi_letter_fill", app="libreoffice_writer", instruction=instr,
                files={xl: xlsx_bytes([("Sheet1", rows)]), template: docx_bytes(tpl)},
                check=check, oracle=office_write(out, docx_bytes(want), "docx"), near_misses=nm,
                open_cmd=["libreoffice", "--writer", template], allow=[out], settle_s=8, multi_app=True,
                difficulty={"horizon": "medium", "skills": ["multi_app", "lookup", "save_as"]},
                params={"xlsx": xl, "template": template, "out": out, "customer": who[0], "row": k,
                        "data": digest([rows, tpl])})


FAMILIES = {"multi_csv_report": (csv_report, 96), "multi_letter_fill": (letter_fill, 96)}

