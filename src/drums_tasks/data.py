"""Seeded content for generated tasks: tables, documents, slides, images, small web sites and code."""
from __future__ import annotations

import io

from .core import FIXED_ZIP_TIME, normalize_zip, pick

FIRST = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kira", "Liam", "Maya",
         "Nikhil", "Olga", "Pablo", "Quinn", "Rosa", "Sami", "Tara", "Uma", "Victor", "Wen", "Yusuf", "Zoe"]
LAST = ["Adams", "Brooks", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Hughes", "Ito", "Jensen", "Khan", "Lopez",
        "Moreau", "Novak", "Okafor", "Patel", "Quist", "Rossi", "Silva", "Tanaka", "Ueda", "Varga", "Weber", "Young"]
CITIES = ["Austin", "Berlin", "Cairo", "Denver", "Lisbon", "Madrid", "Nairobi", "Osaka", "Porto", "Quito", "Riga",
          "Seoul", "Toronto", "Utrecht", "Vienna", "Warsaw"]
PRODUCTS = ["Anvil", "Beacon", "Compass", "Dynamo", "Ember", "Falcon", "Glacier", "Harbor", "Iris", "Juniper",
            "Kestrel", "Lantern", "Meadow", "Nimbus", "Orchid", "Pioneer", "Quartz", "Raven", "Summit", "Tundra"]
REGIONS = ["North", "South", "East", "West", "Central"]
WORDS = ["budget", "review", "schedule", "vendor", "invoice", "meeting", "project", "report", "deadline", "contract",
         "release", "customer", "estimate", "summary", "proposal", "timeline", "analysis", "feedback", "inventory"]
TOPICS = ["Quarterly planning", "Office move", "Product launch", "Hiring plan", "Security review", "Budget update",
          "Customer survey", "Vendor selection", "Training program", "Website redesign", "Data migration",
          "Team offsite"]

DATASETS = {
    "sales": {"file": "sales", "cols": ["Region", "Product", "Units", "Price", "Revenue"], "num": ["Units", "Price", "Revenue"]},
    "inventory": {"file": "inventory", "cols": ["Item", "Warehouse", "Stock", "Reorder", "Cost"], "num": ["Stock", "Reorder", "Cost"]},
    "grades": {"file": "grades", "cols": ["Student", "City", "Math", "Science", "English"], "num": ["Math", "Science", "English"]},
    "expenses": {"file": "expenses", "cols": ["Employee", "Category", "Amount", "Days", "Rate"], "num": ["Amount", "Days", "Rate"]},
}
CATS = ["Travel", "Meals", "Lodging", "Supplies", "Software", "Training"]


def table(rng, ds: str, n: int) -> list:
    """Header + n rows of the dataset, values drawn from rng. Numbers are integers (no float round-trip issues)."""
    d = DATASETS[ds]
    rows = [list(d["cols"])]
    used = set()
    for _ in range(n):
        if ds == "sales":
            u, p = rng.randint(3, 400), rng.randint(5, 250)
            r = [pick(rng, REGIONS), pick(rng, PRODUCTS), u, p, u * p]
        elif ds == "inventory":
            name = pick(rng, PRODUCTS) + "-" + str(rng.randint(10, 99))
            while name in used:
                name = pick(rng, PRODUCTS) + "-" + str(rng.randint(10, 99))
            used.add(name)
            r = [name, pick(rng, CITIES), rng.randint(0, 900), rng.randint(20, 200), rng.randint(2, 480)]
        elif ds == "grades":
            name = f"{pick(rng, FIRST)} {pick(rng, LAST)}"
            while name in used:
                name = f"{pick(rng, FIRST)} {pick(rng, LAST)}"
            used.add(name)
            r = [name, pick(rng, CITIES), rng.randint(35, 100), rng.randint(35, 100), rng.randint(35, 100)]
        else:
            name = f"{pick(rng, FIRST)} {pick(rng, LAST)}"
            r = [name, pick(rng, CATS), rng.randint(15, 2400), rng.randint(1, 14), rng.randint(40, 320)]
        rows.append(r)
    return rows


def xlsx_bytes(sheets: list) -> bytes:
    """sheets: [(name, rows)] -> deterministic xlsx bytes (openpyxl, fixed properties and zip times)."""
    import datetime

    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    t = datetime.datetime(*FIXED_ZIP_TIME)
    wb.properties.created = t
    wb.properties.modified = t
    wb.properties.creator = "drums-t6"
    b = io.BytesIO()
    wb.save(b)
    return normalize_zip(b.getvalue())


def csv_bytes(rows: list) -> bytes:
    import csv
    s = io.StringIO()
    w = csv.writer(s, lineterminator="\n")
    for r in rows:
        w.writerow(r)
    return s.getvalue().encode()


def sentence(rng, i: int) -> str:
    a, b, c = pick(rng, WORDS), pick(rng, WORDS), pick(rng, CITIES)
    n = rng.randint(2, 40)
    forms = [f"The {a} for {c} is due in {n} days and the {b} team owns it.",
             f"We agreed to move the {a} review to {c} after {n} open items are closed.",
             f"{pick(rng, FIRST)} will send the {a} {b} to the {c} office by day {n}.",
             f"Item {i + 1}: the {a} needs a {b} before the {c} visit, about {n} hours of work.",
             f"Our {b} shows {n} percent growth in {c}, so the {a} stays on track."]
    return pick(rng, forms)


def docx_bytes(paragraphs: list, bold: str | None = None, heading: str | None = None) -> bytes:
    """A deterministic .docx: an optional heading paragraph, then paragraphs; `bold` = a phrase already bold."""
    import datetime

    import docx
    d = docx.Document()
    if heading:
        d.add_heading(heading, level=1)
    for p in paragraphs:
        para = d.add_paragraph()
        if bold and bold in p:
            i = p.index(bold)
            para.add_run(p[:i])
            para.add_run(bold).bold = True
            para.add_run(p[i + len(bold):])
        else:
            para.add_run(p)
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


def pptx_bytes(slides: list) -> bytes:
    """slides: [(title, [bullets])] -> deterministic .pptx (title + content layout)."""
    import datetime

    import pptx
    prs = pptx.Presentation()
    for title, bullets in slides:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        body = s.placeholders[1].text_frame
        for i, bl in enumerate(bullets):
            if i == 0:
                body.text = bl
            else:
                body.add_paragraph().text = bl
    t = datetime.datetime(*FIXED_ZIP_TIME)
    cp = prs.core_properties
    cp.created = t
    cp.modified = t
    cp.author = "drums-t6"
    cp.last_modified_by = "drums-t6"
    cp.revision = 1
    b = io.BytesIO()
    prs.save(b)
    return normalize_zip(b.getvalue())


def image(rng, w: int, h: int):
    """A PIL image with asymmetric shapes (so every rotation and flip is distinguishable)."""
    from PIL import Image, ImageDraw
    bg = tuple(rng.randint(30, 225) for _ in range(3))
    im = Image.new("RGB", (w, h), bg)
    dr = ImageDraw.Draw(im)
    for _ in range(rng.randint(3, 6)):
        x0, y0 = rng.randint(0, w - 10), rng.randint(0, h - 10)
        x1, y1 = rng.randint(x0 + 5, min(w, x0 + w // 2)), rng.randint(y0 + 5, min(h, y0 + h // 2))
        col = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            dr.rectangle([x0, y0, x1, y1], fill=col)
        else:
            dr.ellipse([x0, y0, x1, y1], fill=col)
    # a fixed corner marker breaks every symmetry
    dr.rectangle([0, 0, w // 6, h // 8], fill=(250, 20, 20))
    dr.rectangle([w - w // 10, h - h // 5, w - 1, h - 1], fill=(20, 20, 250))
    return im


def png_bytes(im) -> bytes:
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=False, compress_level=6)
    return b.getvalue()


def jpeg_bytes(im, q=92) -> bytes:
    b = io.BytesIO()
    im.convert("RGB").save(b, format="JPEG", quality=q)
    return b.getvalue()


def html_page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
            f"<body><h1>{title}</h1>{body}</body></html>").encode()
