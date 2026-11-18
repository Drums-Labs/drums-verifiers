"""Multi-app workflows: the answer comes from one app and goes into another (or into the shell's filesystem)."""
from __future__ import annotations

import io
import zipfile

from ..core import FIXED_ZIP_TIME, HOME, Spec, office_write, phrase, pick, sha256, sh, sleep, write_file_steps
from ..data import DATASETS, PRODUCTS, csv_bytes, docx_bytes, html_page, sentence, table, xlsx_bytes
from .apps import BASE, SITE, chrome_open, site_setup
from .lib import digest


def calc_to_text(rng, idx):
    for _ in range(50):
        ds = pick(rng, ["grades", "inventory"])
        rows = table(rng, ds, rng.randint(8, 15))
        label = rows[0][0]
        num = pick(rng, DATASETS[ds]["num"])
        ci = rows[0].index(num)
        vals = [r[ci] for r in rows[1:]]
        labels = [r[0] for r in rows[1:]]
        if len(set(vals)) == len(vals) and len(set(labels)) == len(labels):
            break
    hi = rng.random() < 0.5
    order = sorted(rows[1:], key=lambda r: r[ci], reverse=hi)
    ans = order[0][0]
    path = f"{HOME}/Documents/{ds}.xlsx"
    out = f"{HOME}/Desktop/answer.txt"
    instr = phrase(rng, [
        "Look at {f}: which {l} has the {w} {n}? Write only that {l} into a new text file {o}.",
        "Open {f} in Calc, find the row with the {w} value of {n}, and put its {l} (just the text) in {o}.",
        "From the spreadsheet {f}, take the {l} of the row whose {n} is the {w}, and save it to {o}."],
        f=path.replace(HOME, "~"), l=label, w="highest" if hi else "lowest", n=num, o=out.replace(HOME, "~"))
    return Spec(family="multi_calc_to_text", app="libreoffice_calc", instruction=instr,
                files={path: xlsx_bytes([("Sheet1", rows)])},
                check={"kind": "text_file", "path": out, "mode": "exact_strip", "expected": ans},
                oracle=write_file_steps(out, (ans + "\n").encode()),
                near_misses=[("second_place", write_file_steps(out, (order[1][0] + "\n").encode())),
                             ("opposite_end", write_file_steps(out, (order[-1][0] + "\n").encode())),
                             ("value_not_label", write_file_steps(out, (f"{order[0][ci]}\n").encode()))],
                open_cmd=["libreoffice", "--calc", path], allow=[out], settle_s=8, multi_app=True,
                params={"dataset": ds, "column": num, "highest": hi, "data": digest(rows)})


def web_to_csv(rng, idx):
    rows = [["Product", "Price"]] + [[p, rng.randint(3, 400)] for p in rng.sample(PRODUCTS, rng.randint(4, 8))]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows[1:])
    files = {f"{SITE}/index.html": html_page("Price list", f"<table border=1><tr><th>Product</th><th>Price</th></tr>{trs}</table>")}
    out = f"{HOME}/Documents/prices.csv"
    instr = phrase(rng, [
        "The web page open in Chrome shows a price table. Save it as {o}: a CSV with the header Product,Price and one line per row of the table.",
        "Copy the table from the open page into a comma-separated file {o} (header Product,Price, rows in page order).",
        "Create {o} containing the Product and Price table shown in the browser, header first, same order."],
        o=out.replace(HOME, "~"))
    wrong = [list(r) for r in rows]
    wrong[1][1] = wrong[1][1] + 1
    return Spec(family="multi_web_to_csv", app="chrome", instruction=instr, files=files,
                check={"kind": "csv_table", "path": out, "rows": rows},
                oracle=write_file_steps(out, csv_bytes(rows)),
                near_misses=[("missing_row", write_file_steps(out, csv_bytes(rows[:-1]))),
                             ("wrong_value", write_file_steps(out, csv_bytes(wrong))),
                             ("no_header", write_file_steps(out, csv_bytes(rows[1:])))],
                extra_setup=site_setup(), allow=[out], settle_s=6, multi_app=True, params={"rows": len(rows) - 1})


def doc_to_sheet(rng, idx):
    items = [[p, rng.randint(1, 60)] for p in rng.sample(PRODUCTS, rng.randint(3, 7))]
    paras = ["Order list"] + [f"{a}: {b}" for a, b in items]
    doc = f"{HOME}/Documents/order_{rng.randint(1, 99):02d}.docx"
    out = f"{HOME}/Documents/order.xlsx"
    rows = [["Item", "Quantity"]] + items
    instr = phrase(rng, [
        "The document {d} lists items with quantities. Put them into a new spreadsheet {o}: header Item and Quantity in row 1, then one row per item in the same order.",
        "Create {o} from the list in {d}: column A Item, column B Quantity (numbers), a header row, same order.",
        "Turn the order list in {d} into the workbook {o} with the columns Item and Quantity."],
        d=doc.replace(HOME, "~"), o=out.replace(HOME, "~"))
    swapped = [["Quantity", "Item"]] + [[b, a] for a, b in items]
    return Spec(family="multi_doc_to_sheet", app="libreoffice_writer", instruction=instr, files={doc: docx_bytes(paras)},
                check={"kind": "xlsx_table", "path": out, "sheet": None, "rows": rows},
                oracle=office_write(out, xlsx_bytes([("Sheet1", rows)]), "xlsx"),
                near_misses=[("swapped_columns", office_write(out, xlsx_bytes([("Sheet1", swapped)]), "xlsx")),
                             ("missing_header", office_write(out, xlsx_bytes([("Sheet1", items)]), "xlsx")),
                             ("missing_item", office_write(out, xlsx_bytes([("Sheet1", rows[:-1])]), "xlsx"))],
                open_cmd=["libreoffice", "--writer", doc], allow=[out], settle_s=8, multi_app=True,
                params={"items": len(items), "doc": doc, "data": digest(items)})


def sheet_to_doc(rng, idx):
    rows = table(rng, "expenses", rng.randint(5, 12))
    total = sum(r[2] for r in rows[1:])
    xl = f"{HOME}/Documents/expenses.xlsx"
    doc = f"{HOME}/Documents/expense_report.docx"
    paras = [sentence(rng, i) for i in range(rng.randint(2, 4))]
    line = f"Total amount: {total}"
    instr = phrase(rng, [
        "Add a last paragraph to {d} that reads \"Total amount: N\", where N is the sum of the Amount column in {x}. Save the document.",
        "Compute the total of Amount in {x} and append the line Total amount: <that number> to {d} as its final paragraph; save.",
        "Finish {d}: its last paragraph must be 'Total amount: ' followed by the total Amount from {x}. Save it."],
        d=doc.replace(HOME, "~"), x=xl.replace(HOME, "~"))
    return Spec(family="multi_sheet_to_doc", app="libreoffice_writer", instruction=instr,
                files={xl: xlsx_bytes([("Sheet1", rows)]), doc: docx_bytes(paras)},
                check={"kind": "docx_paragraphs", "path": doc, "paragraphs": paras + [line]},
                oracle=office_write(doc, docx_bytes(paras + [line]), "docx"),
                near_misses=[("wrong_total", office_write(doc, docx_bytes(paras + [f"Total amount: {total - rows[-1][2]}"]), "docx")),
                             ("count_not_sum", office_write(doc, docx_bytes(paras + [f"Total amount: {len(rows) - 1}"]), "docx"))],
                open_cmd=["libreoffice", "--writer", doc], allow=[doc], settle_s=8, multi_app=True,
                params={"rows": len(rows) - 1, "total": total, "data": digest([rows, paras])})


def download_extract(rng, idx):
    names = rng.sample(["a.txt", "b.txt", "readme.md", "data.csv", "notes.txt", "list.txt"], rng.randint(2, 4))
    members = {n: (sentence(rng, i) + "\n").encode() for i, n in enumerate(names)}
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(zipfile.ZipInfo(n, FIXED_ZIP_TIME), members[n])
    arc = f"bundle_{rng.randint(10, 99)}.zip"
    files = {f"{SITE}/files/{arc}": b.getvalue(),
             f"{SITE}/index.html": html_page("Bundle", f"<p><a href='files/{arc}'>{arc}</a></p>")}
    dest = f"{HOME}/Documents/extracted"
    instr = phrase(rng, [
        "Download {a} from the page open in Chrome and extract its files into {d}.",
        "Get the archive {a} linked on the current page and unpack it so its files are in {d} (no sub-folder).",
        "Download {a} with the browser, then extract everything in it directly into the folder {d}."],
        a=arc, d=dest.replace(HOME, "~"))
    want = {f"{dest}/{n}": sha256(v) for n, v in members.items()}
    dl = f"{HOME}/Downloads/{arc}"
    return Spec(family="multi_download_extract", app="chrome", instruction=instr, files=files,
                check={"kind": "files_state", "must_exist": want, "listing": {dest: sorted(names)}},
                oracle=[chrome_open(f"{BASE}/files/{arc}"), sleep(4),
                        sh(f"mkdir -p '{dest}' && cd '{dest}' && unzip -q -o '{dl}'")],
                near_misses=[("downloaded_only", [chrome_open(f"{BASE}/files/{arc}"), sleep(4)]),
                             ("nested_folder", [chrome_open(f"{BASE}/files/{arc}"), sleep(4),
                                                sh(f"mkdir -p '{dest}/bundle' && cd '{dest}/bundle' && unzip -q -o '{dl}'")]),
                             ("missing_member", [sh(f"mkdir -p '{dest}'")] + [s for n in names[:-1] for s in write_file_steps(f"{dest}/{n}", members[n])])],
                extra_setup=site_setup(), allow=[f"{dest}", f"{dest}/*", f"{HOME}/Downloads/*"], settle_s=6, multi_app=True,
                params={"members": len(names)})


FAMILIES = {"multi_calc_to_text": (calc_to_text, 40), "multi_web_to_csv": (web_to_csv, 30),
            "multi_doc_to_sheet": (doc_to_sheet, 30), "multi_sheet_to_doc": (sheet_to_doc, 30),
            "multi_download_extract": (download_extract, 30)}
