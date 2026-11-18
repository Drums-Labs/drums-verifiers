"""LibreOffice Writer and Impress families."""
from __future__ import annotations

from ..core import HOME, Spec, lo_convert, office_write, phrase, pick, write_file_steps
from ..data import TOPICS, WORDS, docx_bytes, pptx_bytes, sentence
from .lib import digest

DIRS = [f"{HOME}/Documents", f"{HOME}/Desktop"]
NAMES = ["memo", "minutes", "notes", "plan", "letter", "brief", "draft", "update", "agenda", "summary"]


def _doc(rng, n=None):
    n = n or rng.randint(4, 8)
    paras = []
    while len(paras) < n:
        s = sentence(rng, len(paras))
        if s not in paras:
            paras.append(s)
    path = f"{pick(rng, DIRS)}/{pick(rng, NAMES)}_{rng.randint(1, 99):02d}.docx"
    return paras, path


def _open(path):
    return ["libreoffice", "--writer", path]


def _w(path, paras, bold=None, heading=None):
    return office_write(path, docx_bytes(paras, bold=bold, heading=heading), "docx")


def _check(path, paras, heading=None):
    return {"kind": "docx_paragraphs", "path": path, "paragraphs": ([heading] if heading else []) + paras}


def replace_word(rng, idx):
    for _ in range(100):
        paras, path = _doc(rng, rng.randint(5, 8))
        text = " ".join(paras)
        counts = {w: text.count(w) for w in WORDS if text.count(w) >= 2}
        if counts:
            break
    old = pick(rng, sorted(counts))
    new = pick(rng, [w for w in ["plan", "draft", "audit", "roadmap", "ticket", "order", "memo"] if w not in text])
    want = [p.replace(old, new) for p in paras]
    first = []
    done = False
    for p in paras:
        if not done and old in p:
            first.append(p.replace(old, new, 1))
            done = True
        else:
            first.append(p)
    upper = [p.replace(old, new.capitalize()) for p in paras]
    instr = phrase(rng, [
        "In {f}, replace every occurrence of the word \"{o}\" with \"{n}\" and save the document in the same format.",
        "Change all \"{o}\" to \"{n}\" throughout {f} (lower case, as written) and save it.",
        "The document {f} uses \"{o}\" where it should say \"{n}\". Fix every instance and save."],
        f=path.replace(HOME, "~"), o=old, n=new)
    return Spec(family="writer_replace", app="libreoffice_writer", instruction=instr, files={path: docx_bytes(paras)},
                check=_check(path, want), oracle=_w(path, want),
                near_misses=[("first_only", _w(path, first)), ("wrong_case", _w(path, upper)),
                             ("replaced_other_word", _w(path, [p.replace(old, old + "s") for p in paras]))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"old": old, "new": new, "n": counts[old], "path": path,
                                                                         "text": digest(paras)})


def append_para(rng, idx):
    paras, path = _doc(rng)
    new = sentence(rng, 99)
    while new in paras:
        new = sentence(rng, 98)
    typo = new.replace(" the ", " teh ", 1) if " the " in new else new[:-1]
    instr = phrase(rng, [
        "Add this sentence as a new last paragraph of {f}, exactly as written: \"{s}\". Save the document.",
        "At the end of {f}, add a new paragraph that says: {s} -- then save in the same format.",
        "Append the paragraph \"{s}\" to the end of {f} and save it."], f=path.replace(HOME, "~"), s=new)
    return Spec(family="writer_append", app="libreoffice_writer", instruction=instr, files={path: docx_bytes(paras)},
                check=_check(path, paras + [new]), oracle=_w(path, paras + [new]),
                near_misses=[("inserted_first", _w(path, [new] + paras)), ("typo", _w(path, paras + [typo])),
                             ("replaced_last", _w(path, paras[:-1] + [new]))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"n": len(paras), "path": path, "new": new,
                                                                         "text": digest(paras)})


def delete_para(rng, idx):
    paras, path = _doc(rng, rng.randint(5, 8))
    k = rng.randrange(len(paras))
    start = " ".join(paras[k].split()[:5])
    want = paras[:k] + paras[k + 1:]
    j = (k + 1) % len(paras)
    other = paras[:j] + paras[j + 1:]
    half = list(paras)
    half[k] = " ".join(paras[k].split()[:5])
    instr = phrase(rng, [
        "Delete the paragraph of {f} that begins with \"{s}\" (the whole paragraph) and save.",
        "Remove the whole paragraph starting \"{s}...\" from {f}; keep the rest unchanged and save.",
        "In {f}, the paragraph beginning \"{s}\" is no longer needed. Delete it completely and save the file."],
        f=path.replace(HOME, "~"), s=start)
    return Spec(family="writer_delete_para", app="libreoffice_writer", instruction=instr, files={path: docx_bytes(paras)},
                check=_check(path, want), oracle=_w(path, want),
                near_misses=[("deleted_other", _w(path, other)), ("partial_delete", _w(path, half)),
                             ("deleted_two", _w(path, [p for i, p in enumerate(paras) if i not in (k, j)]))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"k": k, "n": len(paras), "path": path, "text": digest(paras)})


def bold_phrase(rng, idx):
    paras, path = _doc(rng)
    k = rng.randrange(len(paras))
    words = paras[k].rstrip(".").split()
    i = rng.randrange(max(1, len(words) - 3))
    phrase_ = " ".join(words[i:i + rng.randint(2, 3)])
    if " ".join(paras).count(phrase_) != 1:
        phrase_ = " ".join(words[:4])
    others = [p for p in paras if p != paras[k]]
    other_phrase = " ".join(others[0].split()[:3]) if others else paras[k].split()[0]
    instr = phrase(rng, [
        "Make the words \"{p}\" in {f} bold (only those words) and save the document.",
        "In {f}, format the phrase \"{p}\" in bold. Nothing else changes. Save.",
        "Bold the text \"{p}\" in the document {f}, then save it in the same format."],
        p=phrase_, f=path.replace(HOME, "~"))
    from ..data import docx_bytes as db
    import io

    def italic(paras2, ph):
        import docx
        from ..core import FIXED_ZIP_TIME, normalize_zip
        import datetime
        d = docx.Document()
        for p in paras2:
            para = d.add_paragraph()
            if ph in p:
                a = p.index(ph)
                para.add_run(p[:a])
                para.add_run(ph).italic = True
                para.add_run(p[a + len(ph):])
            else:
                para.add_run(p)
        t = datetime.datetime(*FIXED_ZIP_TIME)
        d.core_properties.created = t
        d.core_properties.modified = t
        b = io.BytesIO()
        d.save(b)
        return office_write(path, normalize_zip(b.getvalue()), "docx")
    return Spec(family="writer_bold", app="libreoffice_writer", instruction=instr, files={path: db(paras)},
                check={"kind": "docx_bold", "path": path, "paragraphs": paras, "bold": phrase_},
                oracle=_w(path, paras, bold=phrase_),
                near_misses=[("italic_instead", italic(paras, phrase_)), ("bolded_other", _w(path, paras, bold=other_phrase)),
                             ("bolded_whole_paragraph", _w(path, paras, bold=paras[k]))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"phrase": phrase_, "path": path, "text": digest(paras)})


def export_pdf(rng, idx):
    paras, path = _doc(rng)
    d, name = path.rsplit("/", 1)
    stem = name[:-5]
    out = f"{d}/{stem}.pdf"
    other_paras, _ = _doc(rng)
    other_path = f"{d}/{stem}_old.docx"
    instr = phrase(rng, [
        "Export {f} as a PDF named {n} in the same folder.",
        "Save a PDF version of the document {f} next to it as {n}.",
        "Create {o} from the document {f} (File > Export as PDF or any other way)."],
        f=path.replace(HOME, "~"), n=f"{stem}.pdf", o=out.replace(HOME, "~"))
    other_dir = f"{HOME}/Desktop" if d.endswith("Documents") else f"{HOME}/Documents"
    stage = f"/tmp/t6pdf-{idx}"
    wrong_doc = [{"type": "execute", "command": ["mkdir", "-p", stage], "shell": False},
                 {"type": "execute", "command": ["cp", other_path, f"{stage}/{stem}.docx"], "shell": False},
                 lo_convert(f"{stage}/{stem}.docx", "pdf", d)]
    return Spec(family="writer_export_pdf", app="libreoffice_writer", instruction=instr,
                files={path: docx_bytes(paras), other_path: docx_bytes(other_paras)},
                check={"kind": "pdf_text", "path": out, "contains": [paras[0], paras[-1]], "absent": [other_paras[0]]},
                oracle=[lo_convert(path, "pdf", d)],
                near_misses=[("wrong_document", wrong_doc), ("wrong_folder", [lo_convert(path, "pdf", other_dir)]),
                             ("not_a_pdf", write_file_steps(out, "\n".join(paras).encode()))],
                open_cmd=_open(path), allow=[out], settle_s=8, params={"n": len(paras), "path": path,
                                                                        "text": digest([paras, other_paras])})


def heading(rng, idx):
    paras, path = _doc(rng, rng.randint(3, 6))
    old = pick(rng, TOPICS)
    new = pick(rng, [t for t in TOPICS if t != old]) + " " + str(rng.randint(2025, 2027))
    instr = phrase(rng, [
        "Change the heading of {f} from \"{o}\" to \"{n}\" and save.",
        "The title line of {f} reads \"{o}\". Make it read \"{n}\" instead, then save the document.",
        "Rename the document heading in {f} to \"{n}\" (it currently says \"{o}\"). Save it."],
        f=path.replace(HOME, "~"), o=old, n=new)
    return Spec(family="writer_heading", app="libreoffice_writer", instruction=instr,
                files={path: docx_bytes(paras, heading=old)},
                check=_check(path, paras, heading=new), oracle=_w(path, paras, heading=new),
                near_misses=[("appended_not_replaced", _w(path, paras, heading=f"{old} {new}")),
                             ("body_changed_instead", _w(path, [new] + paras[1:], heading=old)),
                             ("new_line_added_old_kept", _w(path, [new] + paras, heading=old))],
                open_cmd=_open(path), allow=[path], settle_s=8, params={"old": old, "new": new, "path": path, "text": digest(paras)})


# ------------------------------------------------------------------------------------------------ Impress

def _deck(rng, n=None):
    n = n or rng.randint(3, 6)
    titles = rng.sample(TOPICS, n)
    slides = [(t, [sentence(rng, i) for i in range(rng.randint(1, 3))]) for t in titles]
    path = f"{pick(rng, DIRS)}/{pick(rng, ['deck', 'slides', 'talk', 'pitch', 'review'])}_{rng.randint(1, 99):02d}.pptx"
    return slides, path


def _iopen(path):
    return ["libreoffice", "--impress", path]


def _p(path, slides):
    return office_write(path, pptx_bytes(slides), "pptx")


def _pcheck(path, slides):
    return {"kind": "pptx_titles", "path": path, "titles": [t for t, _ in slides]}


ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]


def slide_title(rng, idx):
    slides, path = _deck(rng)
    k = rng.randrange(len(slides))
    new = pick(rng, ["Next steps", "Open questions", "Key results", "Risks", "Timeline", "Decisions"]) + f" ({rng.randint(1, 9)})"
    want = [(new if i == k else t, b) for i, (t, b) in enumerate(slides)]
    j = (k + 1) % len(slides)
    other = [(new if i == j else t, b) for i, (t, b) in enumerate(slides)]
    typo = [((new.lower() if i == k else t), b) for i, (t, b) in enumerate(slides)]
    instr = phrase(rng, [
        "In the presentation {f}, change the title of the {o} slide (\"{t}\") to \"{n}\" and save.",
        "Retitle slide {k1} of {f} as \"{n}\". Save the presentation in the same format.",
        "The {o} slide of {f} should be titled \"{n}\" instead of \"{t}\". Make the change and save."],
        f=path.replace(HOME, "~"), o=ORD[k], t=slides[k][0], n=new, k1=k + 1)
    return Spec(family="impress_title", app="libreoffice_impress", instruction=instr, files={path: pptx_bytes(slides)},
                check=_pcheck(path, want), oracle=_p(path, want),
                near_misses=[("wrong_slide", _p(path, other)), ("wrong_case", _p(path, typo)),
                             ("title_appended", _p(path, [((t + " " + new) if i == k else t, b) for i, (t, b) in enumerate(slides)]))],
                open_cmd=_iopen(path), allow=[path], settle_s=9, params={"k": k, "n": len(slides), "path": path, "title": new,
                                                                          "slides": digest(slides)})


def delete_slide(rng, idx):
    slides, path = _deck(rng, rng.randint(3, 6))
    k = rng.randrange(len(slides))
    want = slides[:k] + slides[k + 1:]
    j = (k + 1) % len(slides)
    other = slides[:j] + slides[j + 1:]
    cleared = [(("" if i == k else t), ([] if i == k else b)) for i, (t, b) in enumerate(slides)]
    instr = phrase(rng, [
        "Delete the slide titled \"{t}\" from {f} and save the presentation.",
        "Remove slide {k1} (\"{t}\") from the deck {f}. Save it.",
        "The deck {f} should no longer contain the \"{t}\" slide. Delete that slide and save."],
        f=path.replace(HOME, "~"), t=slides[k][0], k1=k + 1)
    return Spec(family="impress_delete_slide", app="libreoffice_impress", instruction=instr,
                files={path: pptx_bytes(slides)}, check=_pcheck(path, want), oracle=_p(path, want),
                near_misses=[("deleted_other", _p(path, other)), ("cleared_not_deleted", _p(path, cleared)),
                             ("deleted_two", _p(path, [s for i, s in enumerate(slides) if i not in (k, j)]))],
                open_cmd=_iopen(path), allow=[path], settle_s=9, params={"k": k, "n": len(slides), "path": path,
                                                                          "slides": digest(slides)})


def add_slide(rng, idx):
    slides, path = _deck(rng, rng.randint(2, 5))
    new = pick(rng, ["Thank you", "Questions", "Appendix", "Summary", "Contacts", "Roadmap"]) + f" {rng.randint(1, 99)}"
    at_end = rng.random() < 0.6
    want = slides + [(new, [])] if at_end else [(new, [])] + slides
    wrong_pos = [(new, [])] + slides if at_end else slides + [(new, [])]
    instr = phrase(rng, [
        "Add a new slide {w} of {f} with the title \"{n}\" and save.",
        "Insert a slide titled \"{n}\" {w} of the presentation {f}. Save it in the same format.",
        "The deck {f} needs one more slide, {w}, titled \"{n}\". Add it and save."],
        f=path.replace(HOME, "~"), n=new, w="at the end" if at_end else "at the beginning")
    return Spec(family="impress_add_slide", app="libreoffice_impress", instruction=instr,
                files={path: pptx_bytes(slides)}, check=_pcheck(path, want), oracle=_p(path, want),
                near_misses=[("wrong_position", _p(path, wrong_pos)),
                             ("title_typo", _p(path, [(t + "." if t == new else t, b) for t, b in want])),
                             ("first_title_replaced", _p(path, [(new, slides[0][1])] + slides[1:]))],
                open_cmd=_iopen(path), allow=[path], settle_s=9, params={"at_end": at_end, "n": len(slides), "path": path, "title": new,
                                                                          "slides": digest(slides)})


FAMILIES = {"writer_replace": (replace_word, 60), "writer_append": (append_para, 50), "writer_delete_para": (delete_para, 50),
            "writer_bold": (bold_phrase, 50), "writer_export_pdf": (export_pdf, 40), "writer_heading": (heading, 40),
            "impress_title": (slide_title, 50), "impress_delete_slide": (delete_slide, 40), "impress_add_slide": (add_slide, 40)}
