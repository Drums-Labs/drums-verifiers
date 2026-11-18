"""File-manager / OS families: rename, move by extension, delete by pattern, folder tree, zip, count, find-and-copy,
concatenate. The file manager (nautilus) is open on the working folder; a shell works as well."""
from __future__ import annotations

import io
import zipfile

from ..core import FIXED_ZIP_TIME, HOME, Spec, phrase, pick, py, sha256, sh, write_file_steps
from ..data import CITIES, PRODUCTS, WORDS, sentence
from .lib import seeded

BASES = [f"{HOME}/Documents", f"{HOME}/Desktop", f"{HOME}/Downloads"]
EXTS = ["txt", "csv", "md", "log", "json"]


def _content(rng, ext, i):
    if ext == "csv":
        return f"id,city,value\n{i},{pick(rng, CITIES)},{rng.randint(1, 999)}\n".encode()
    if ext == "json":
        return ('{"id": %d, "product": "%s", "qty": %d}\n' % (i, pick(rng, PRODUCTS), rng.randint(1, 99))).encode()
    return (sentence(rng, i) + "\n").encode()


def _folder(rng, n, exts=None):
    base = pick(rng, BASES)
    d = f"{base}/{pick(rng, ['work', 'files', 'inbox', 'export', 'batch', 'archive'])}_{rng.randint(10, 99)}"
    files = {}
    names = set()
    while len(files) < n:
        ext = pick(rng, exts or EXTS)
        nm = f"{pick(rng, WORDS)}_{rng.randint(1, 60)}.{ext}"
        if nm in names:
            continue
        names.add(nm)
        files[f"{d}/{nm}"] = _content(rng, ext, len(files))
    return d, files


def _open(d):
    return ["nautilus", d]


def _mv(a, b):
    return {"type": "execute", "command": ["mv", a, b], "shell": False}


def _cp(a, b):
    return {"type": "execute", "command": ["cp", a, b], "shell": False}


def _rm(*p):
    return {"type": "execute", "command": ["rm", "-rf", *p], "shell": False}


def _mkdir(*p):
    return {"type": "execute", "command": ["mkdir", "-p", *p], "shell": False}


def _state(d, files, extra_absent=()):
    return {"kind": "files_state", "must_exist": {p: sha256(b) for p, b in files.items()},
            "must_absent": list(extra_absent), "listing": _listing(d, files)}


def _listing(d, files):
    out = {}
    for p in files:
        parent, name = p.rsplit("/", 1)
        out.setdefault(parent, []).append(name)
    out.setdefault(d, [])
    for parent in list(out):
        # a sub-folder shows up in its parent's listing
        if parent != d and parent.startswith(d + "/"):
            rel = parent[len(d) + 1:].split("/")[0]
            if rel not in out[d]:
                out[d].append(rel)
    return out


def rename(rng, idx):
    d, files = _folder(rng, rng.randint(3, 7))
    src = pick(rng, sorted(files))
    ext = src.rsplit(".", 1)[1]
    new = f"{pick(rng, ['final', 'approved', 'v2', 'clean', 'shared'])}_{pick(rng, WORDS)}.{ext}"
    dst = f"{d}/{new}"
    want = {(dst if p == src else p): b for p, b in files.items()}
    other = pick(rng, [p for p in sorted(files) if p != src] or [src])
    wrong = {(f"{d}/{new}" if p == other else p): b for p, b in files.items()}
    instr = phrase(rng, [
        "Rename the file {s} in {d} to {n}.", "In the folder {d}, change the name of {s} to {n}.",
        "The file {d}/{s} should be called {n}. Rename it (same folder)."],
        s=src.rsplit("/", 1)[1], d=d.replace(HOME, "~"), n=new)
    return Spec(family="files_rename", app="files", instruction=instr, files=files,
                check=_state(d, want, [src]), oracle=[_mv(src, dst)],
                near_misses=[("renamed_other", [_mv(other, dst)] if other != src else [_cp(src, dst)]),
                             ("copied_not_renamed", [_cp(src, dst)]),
                             ("wrong_extension", [_mv(src, dst + ".bak")])],
                open_cmd=_open(d), allow=[f"{d}/*"], settle_s=3, params={"n": len(files), "src": src, "new": new, **seeded(files)})


def move_ext(rng, idx):
    d, files = _folder(rng, rng.randint(5, 10))
    exts = sorted({p.rsplit(".", 1)[1] for p in files})
    ext = pick(rng, exts)
    sub = f"{d}/{ext}_files"
    want = {(f"{sub}/{p.rsplit('/', 1)[1]}" if p.endswith("." + ext) else p): b for p, b in files.items()}
    movers = [p for p in sorted(files) if p.endswith("." + ext)]
    oracle = [_mkdir(sub)] + [_mv(p, sub + "/") for p in movers]
    partial = [_mkdir(sub)] + [_mv(p, sub + "/") for p in movers[:-1]] if len(movers) > 1 else [_mkdir(sub)]
    other = pick(rng, [e for e in exts if e != ext] or [ext])
    wrong = [_mkdir(sub)] + [_mv(p, sub + "/") for p in sorted(files) if p.endswith("." + other)]
    copied = [_mkdir(sub)] + [_cp(p, sub + "/") for p in movers]
    instr = phrase(rng, [
        "In {d}, create a folder named {s} and move every .{e} file into it.",
        "Move all the .{e} files in {d} into a new sub-folder called {s}; leave the other files where they are.",
        "Tidy up {d}: every file ending in .{e} goes into the folder {d}/{s} (create it)."],
        d=d.replace(HOME, "~"), s=sub.rsplit("/", 1)[1], e=ext)
    return Spec(family="files_move_ext", app="files", instruction=instr, files=files,
                check={**_state(d, want, movers), "dirs": [sub]}, oracle=oracle,
                near_misses=[("one_left_behind", partial), ("wrong_extension", wrong), ("copied_not_moved", copied)],
                open_cmd=_open(d), allow=[f"{d}/*"], settle_s=3, params={"ext": ext, "n": len(files), "moved": len(movers),
                                                                       "sub": sub, **seeded(files)})


def delete_pattern(rng, idx):
    d, files = _folder(rng, rng.randint(5, 10))
    word = pick(rng, sorted({p.rsplit("/", 1)[1].split("_")[0] for p in files}))
    doomed = [p for p in sorted(files) if p.rsplit("/", 1)[1].startswith(word + "_")]
    want = {p: b for p, b in files.items() if p not in doomed}
    others = [p for p in sorted(files) if p not in doomed]
    instr = phrase(rng, [
        "Delete every file in {d} whose name starts with \"{w}_\". Keep all other files.",
        "Remove the files beginning with {w}_ from the folder {d}.",
        "In {d}, get rid of all {w}_* files; nothing else should be deleted."], d=d.replace(HOME, "~"), w=word)
    nm = [("one_left", [_rm(*doomed[:-1])] if len(doomed) > 1 else [py("pass")]),
          ("deleted_extra", [_rm(*doomed, others[0])] if others else [_rm(*doomed), py("pass")]),
          ("moved_not_deleted", [_mkdir(f"{d}/trash")] + [_mv(p, f"{d}/trash/") for p in doomed])]
    return Spec(family="files_delete_pattern", app="files", instruction=instr, files=files,
                check=_state(d, want, doomed), oracle=[_rm(*doomed)], near_misses=nm,
                open_cmd=_open(d), allow=[f"{d}/*"], settle_s=3, params={"word": word, "deleted": len(doomed), **seeded(files)})


def mkdir_tree(rng, idx):
    base = pick(rng, BASES)
    root = f"{base}/{pick(rng, ['project', 'client', 'course', 'trip', 'event'])}_{pick(rng, CITIES).lower()}"
    subs = rng.sample(["docs", "data", "images", "drafts", "final", "notes", "src", "tests", "budget", "photos"],
                      rng.randint(2, 4))
    nested = pick(rng, subs)
    deep = pick(rng, ["2025", "2026", "archive", "old", "raw"])
    dirs = [root] + [f"{root}/{s}" for s in subs] + [f"{root}/{nested}/{deep}"]
    listing = {root: sorted(subs)}
    listing.update({f"{root}/{s}": ([deep] if s == nested else []) for s in subs})
    listing[f"{root}/{nested}/{deep}"] = []
    instr = phrase(rng, [
        "Create the folder {r} with the sub-folders {s}, and inside {n} one more folder named {x}. Nothing else.",
        "Make this folder structure: {r} containing {s}; then {n}/{x} inside it. The folders stay empty.",
        "Set up {r} with empty folders {s}, plus {n}/{x}."],
        r=root.replace(HOME, "~"), s=", ".join(subs), n=nested, x=deep)
    return Spec(family="files_mkdir_tree", app="files", instruction=instr, files={},
                check={"kind": "files_state", "dirs": dirs, "listing": listing},
                oracle=[_mkdir(*dirs)],
                near_misses=[("missing_nested", [_mkdir(*dirs[:-1])]),
                             ("nested_in_wrong_folder", [_mkdir(*dirs[:-1], f"{root}/{deep}")]),
                             ("extra_file", [_mkdir(*dirs), py("open(__import__('sys').argv[1],'w').write('x')", f"{root}/notes.txt")])],
                open_cmd=_open(base), allow=[f"{root}", f"{root}/*"], settle_s=3, params={"subs": subs, "root": root, "nested": nested, "deep": deep})


def _zip_bytes(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.writestr(zipfile.ZipInfo(p.rsplit("/", 1)[1], FIXED_ZIP_TIME), files[p])
    return b.getvalue()


def zip_folder(rng, idx):
    d, files = _folder(rng, rng.randint(3, 6))
    out = f"{d.rsplit('/', 1)[0]}/{d.rsplit('/', 1)[1]}.zip"
    members = {p.rsplit("/", 1)[1]: sha256(b) for p, b in files.items()}
    first = sorted(files)[0]
    instr = phrase(rng, [
        "Compress the folder {d} into a zip archive at {o} (all of its files).",
        "Make a zip file {o} that contains every file in {d}.",
        "Create the archive {o} holding the files of {d}. Keep the folder itself."],
        d=d.replace(HOME, "~"), o=out.replace(HOME, "~"))
    some = {p: b for p, b in files.items() if p != first}
    altered = dict(files)
    altered[first] = files[first] + b"changed\n"
    return Spec(family="files_zip", app="files", instruction=instr, files=files,
                check={"kind": "all", "checks": [{"kind": "zip_members", "path": out, "members": members},
                                                 {"kind": "files_state", "must_exist": {p: sha256(b) for p, b in files.items()}}]},
                oracle=[sh(f"cd '{d.rsplit('/', 1)[0]}' && zip -q -r '{out}' '{d.rsplit('/', 1)[1]}'")],
                near_misses=[("missing_file", write_file_steps(out, _zip_bytes(some))),
                             ("changed_member", write_file_steps(out, _zip_bytes(altered))),
                             ("moved_not_copied", [sh(f"cd '{d.rsplit('/', 1)[0]}' && zip -q -r -m '{out}' '{d.rsplit('/', 1)[1]}'")])],
                open_cmd=_open(d.rsplit("/", 1)[0]), allow=[out], settle_s=3, params={"n": len(files), "out": out, **seeded(files)})


def count_lines(rng, idx):
    d, files = _folder(rng, rng.randint(1, 3), exts=["txt", "log"])
    target = pick(rng, sorted(files))
    body = "".join(sentence(rng, i) + "\n" for i in range(rng.randint(6, 40)))
    files[target] = body.encode()
    n = body.count("\n")
    what = pick(rng, ["lines", "words"])
    val = n if what == "lines" else len(body.split())
    ans = f"{d}/count.txt"
    instr = phrase(rng, [
        "Count the {w} in {f} and write just the number into a new file {a}.",
        "How many {w} does {f} have? Put the number (digits only) in {a}.",
        "Write the number of {w} of the file {f} into {a}."],
        w=what, f=target.replace(HOME, "~"), a=ans.replace(HOME, "~"))
    return Spec(family="files_count", app="files", instruction=instr, files=files,
                check={"kind": "text_file", "path": ans, "mode": "number", "expected": val, "tol": 0},
                oracle=write_file_steps(ans, f"{val}\n".encode()),
                near_misses=[("off_by_one", write_file_steps(ans, f"{val + 1}\n".encode())),
                             ("other_metric", write_file_steps(ans, f"{(len(body.split()) if what == 'lines' else n)}\n".encode())),
                             ("wrong_file_name", write_file_steps(f"{d}/count.text", f"{val}\n".encode()))],
                open_cmd=_open(d), allow=[ans], settle_s=3, params={"what": what, "value": val, "target": target, "answer": ans, **seeded(files)})


def find_copy(rng, idx):
    d, files = _folder(rng, rng.randint(4, 8), exts=["txt", "md", "log"])
    target = pick(rng, sorted(files))
    code = f"{pick(rng, PRODUCTS).upper()}-{rng.randint(1000, 9999)}"
    files[target] = files[target] + f"Reference code: {code}\n".encode()
    dest_dir = f"{HOME}/Desktop/found" if not d.startswith(f"{HOME}/Desktop") else f"{HOME}/Documents/found"
    dst = f"{dest_dir}/{target.rsplit('/', 1)[1]}"
    other = pick(rng, [p for p in sorted(files) if p != target])
    instr = phrase(rng, [
        "One file in {d} mentions the code {c}. Copy that file into {x} (create the folder; keep the original).",
        "Find the file in {d} that contains \"{c}\" and put a copy of it in {x}.",
        "Search {d} for the text {c}; copy the file that has it to the folder {x}, same file name."],
        d=d.replace(HOME, "~"), c=code, x=dest_dir.replace(HOME, "~"))
    return Spec(family="files_find_copy", app="files", instruction=instr, files=files,
                check={"kind": "files_state", "must_exist": {dst: sha256(files[target]), target: sha256(files[target])},
                       "listing": {dest_dir: [dst.rsplit("/", 1)[1]]}},
                oracle=[_mkdir(dest_dir), _cp(target, dest_dir + "/")],
                near_misses=[("copied_wrong_file", [_mkdir(dest_dir), _cp(other, dest_dir + "/")]),
                             ("moved_not_copied", [_mkdir(dest_dir), _mv(target, dest_dir + "/")]),
                             ("copied_all", [_mkdir(dest_dir)] + [_cp(p, dest_dir + "/") for p in sorted(files)])],
                open_cmd=_open(d), allow=[dest_dir, f"{dest_dir}/*"], settle_s=3, params={"n": len(files), "target": target, "code": code, "dest": dest_dir, **seeded(files)})


def concat(rng, idx):
    d, files = _folder(rng, rng.randint(3, 5), exts=["txt"])
    order = sorted(files)
    rng.shuffle(order)
    k = rng.randint(2, len(order))
    parts = order[:k]
    out = f"{d}/combined.txt"
    want = b"".join(files[p] for p in parts)
    instr = phrase(rng, [
        "Create {o} that contains the text of {fs}, in that order, one after another.",
        "Concatenate {fs} (in this order) into a new file {o}.",
        "Join the files {fs} into {o}, keeping their order and their exact contents."],
        o=out.replace(HOME, "~"), fs=", ".join(p.rsplit("/", 1)[1] for p in parts))
    return Spec(family="files_concat", app="files", instruction=instr, files=files,
                check={"kind": "files_state", "must_exist": {out: sha256(want), **{p: sha256(files[p]) for p in files}}},
                oracle=write_file_steps(out, want),
                near_misses=[("reversed_order", write_file_steps(out, b"".join(files[p] for p in reversed(parts)))),
                             ("missing_part", write_file_steps(out, b"".join(files[p] for p in parts[:-1]))),
                             ("extra_blank_lines", write_file_steps(out, b"\n".join(files[p] for p in parts)))],
                open_cmd=_open(d), allow=[out], settle_s=3, params={"k": k, "parts": parts, **seeded(files)})


FAMILIES = {"files_rename": (rename, 50), "files_move_ext": (move_ext, 50), "files_delete_pattern": (delete_pattern, 40),
            "files_mkdir_tree": (mkdir_tree, 30), "files_zip": (zip_folder, 30), "files_count": (count_lines, 40),
            "files_find_copy": (find_copy, 40), "files_concat": (concat, 30)}
