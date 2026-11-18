"""Small helpers shared by the task-library families (medium, multi-app, hidden-state and recovery)."""
from __future__ import annotations

import datetime
import io

from ..core import FIXED_ZIP_TIME, HOME, canonical, normalize_zip, sha256


def tilde(p: str) -> str:
    return p.replace(HOME, "~")


def digest(obj) -> str:
    """A short, stable fingerprint of drawn content (tables, paragraphs, file bytes) for `params`: two variants with
    equal params then have equal content too, so a params match in the split's overlap check is a real duplicate."""
    return sha256(canonical(obj))[:16]


def _fixed(core_properties):
    t = datetime.datetime(*FIXED_ZIP_TIME)
    core_properties.created = t
    core_properties.modified = t
    core_properties.author = "drums-t6"
    core_properties.last_modified_by = "drums-t6"
    core_properties.revision = 1


def docx_sections(sections: list) -> bytes:
    """[(heading, [paragraphs])] -> a deterministic .docx; headings use the Heading 2 style (so Writer's Navigator can
    move a heading with its text). An empty heading writes no heading paragraph."""
    import docx
    d = docx.Document()
    for head, body in sections:
        if head:
            d.add_heading(head, level=2)
        for p in body:
            d.add_paragraph(p)
    _fixed(d.core_properties)
    b = io.BytesIO()
    d.save(b)
    return normalize_zip(b.getvalue())


def xlsx_hidden(rows: list, hidden: list, sheet: str = "Sheet1") -> bytes:
    """One sheet with `rows`; the 1-based row numbers in `hidden` are hidden (row height 0 in the GUI)."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    for r in rows:
        ws.append(r)
    for n in hidden:
        ws.row_dimensions[n].hidden = True
    t = datetime.datetime(*FIXED_ZIP_TIME)
    wb.properties.created = t
    wb.properties.modified = t
    wb.properties.creator = "drums-t6"
    b = io.BytesIO()
    wb.save(b)
    return normalize_zip(b.getvalue())


def seeded(files: dict) -> dict:
    """The lossless part of `params` for a family that seeds files: the file paths and a digest of their content."""
    return {"seeded": sorted(files), "content": digest({p: sha256(b) for p, b in files.items()})}
