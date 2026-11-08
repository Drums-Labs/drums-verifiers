"""Chrome, GIMP and VS Code families."""
from __future__ import annotations

import base64
import io
import json

from ..core import HOME, Spec, gui, phrase, pick, py, setup_launch, setup_sleep, sha256, sh, sleep, write_file_steps
from ..data import CITIES, PRODUCTS, TOPICS, html_page, image, jpeg_bytes, png_bytes, sentence

SITE = f"{HOME}/.t6site"
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
CDP_PORT = 1337


def site_setup(start="index.html"):
    """Serve SITE on 127.0.0.1:8000 (the desktop has no outside network) and open Chrome on the start page."""
    return [setup_launch(["python3", "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "--directory", SITE]),
            setup_sleep(1),
            setup_launch(["google-chrome", f"--remote-debugging-port={CDP_PORT}", "--no-first-run",
                          "--no-default-browser-check", "--password-store=basic", f"{BASE}/{start}"])]


def chrome_open(url):
    return {"type": "execute", "command": ["google-chrome", url], "shell": False}


def _pages(rng, n):
    titles = rng.sample(TOPICS + [f"{c} office" for c in CITIES], n)
    pages = {}
    for i, t in enumerate(titles):
        slug = t.lower().replace(" ", "-") + f"-{rng.randint(10, 99)}"
        pages[f"{slug}.html"] = (t, html_page(t, f"<p>{sentence(rng, i)}</p>"))
    return pages


def _index(pages, extra=""):
    links = "".join(f"<li><a href='{f}'>{t}</a></li>" for f, (t, _) in pages.items())
    return html_page("Team portal", f"<ul>{links}</ul>{extra}")


def open_tabs(rng, idx):
    pages = _pages(rng, rng.randint(3, 6))
    names = list(pages)
    want = rng.sample(names, 2)
    rest = [x for x in names if x not in want]
    files = {f"{SITE}/{f}": b for f, (_, b) in pages.items()}
    files[f"{SITE}/index.html"] = _index(pages)
    instr = phrase(rng, [
        "In Chrome, open the pages \"{a}\" and \"{b}\" from the team portal, each in its own tab, and keep the portal tab open.",
        "Using the links on the open page, open \"{a}\" and \"{b}\" in new browser tabs (keep the start page). Do not open other links.",
        "Open two new tabs in Chrome: one with \"{a}\" and one with \"{b}\", both linked from the current page."],
        a=pages[want[0]][0], b=pages[want[1]][0])
    urls = [f"{BASE}/index.html", f"{BASE}/{want[0]}", f"{BASE}/{want[1]}"]
    return Spec(family="chrome_open_tabs", app="chrome", instruction=instr, files=files,
                check={"kind": "chrome_tabs", "urls": urls, "forbid": [f"{BASE}/{r}" for r in rest], "port": CDP_PORT},
                oracle=[chrome_open(urls[1]), chrome_open(urls[2]), sleep(3)],
                near_misses=[("only_one", [chrome_open(urls[1]), sleep(3)]),
                             ("wrong_page", [chrome_open(urls[1]), chrome_open(f"{BASE}/{rest[0]}"), sleep(3)])],
                extra_setup=site_setup(), redundant=False, roots=None, allow=[], settle_s=6,
                params={"pages": len(pages)})


def _blob(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


def download(rng, idx):
    names = rng.sample([f"{p.lower()}_{c.lower()}" for p in PRODUCTS for c in CITIES[:4]], rng.randint(2, 4))
    blobs = {f"{n}.zip": _blob(rng, rng.randint(2000, 9000)) for n in names}
    target = pick(rng, sorted(blobs))
    other = pick(rng, [b for b in sorted(blobs) if b != target])
    links = "".join(f"<li><a href='files/{f}'>{f}</a></li>" for f in blobs)
    files = {f"{SITE}/files/{f}": b for f, b in blobs.items()}
    files[f"{SITE}/index.html"] = html_page("Downloads", f"<ul>{links}</ul>")
    dst = f"{HOME}/Downloads/{target}"
    instr = phrase(rng, [
        "Download the file {t} from the page open in Chrome. It must end up in ~/Downloads with its original name.",
        "Save {t}, linked on the current web page, to the Downloads folder.",
        "From the downloads page in the browser, get {t} into ~/Downloads (same file name)."], t=target)
    return Spec(family="chrome_download", app="chrome", instruction=instr, files=files,
                check={"kind": "files_state", "must_exist": {dst: sha256(blobs[target])}},
                oracle=[chrome_open(f"{BASE}/files/{target}"), sleep(4)],
                near_misses=[("wrong_file", [chrome_open(f"{BASE}/files/{other}"), sleep(4)]),
                             ("truncated", write_file_steps(dst, blobs[target][:-100])),
                             ("wrong_folder", write_file_steps(f"{HOME}/Desktop/{target}", blobs[target]))],
                extra_setup=site_setup(), allow=[f"{HOME}/Downloads/*"], settle_s=6, params={"n": len(blobs)})


def bookmark(rng, idx):
    pages = _pages(rng, rng.randint(2, 4))
    target = pick(rng, sorted(pages))
    other = pick(rng, [p for p in sorted(pages) if p != target])
    files = {f"{SITE}/{f}": b for f, (_, b) in pages.items()}
    files[f"{SITE}/index.html"] = _index(pages)
    url = f"{BASE}/{target}"
    instr = phrase(rng, [
        "Bookmark the page \"{t}\" (linked from the open page) in Chrome.",
        "Open \"{t}\" from the team portal and add it to your Chrome bookmarks.",
        "Add a Chrome bookmark for the \"{t}\" page."], t=pages[target][0])
    star = [gui("import time; pyautogui.click(960, 540); time.sleep(0.5); pyautogui.hotkey('ctrl', 'd'); time.sleep(2);"
                " pyautogui.press('enter')"), sleep(5)]
    return Spec(family="chrome_bookmark", app="chrome", instruction=instr, files=files,
                check={"kind": "chrome_bookmarks", "urls": [url], "forbid": [f"{BASE}/{other}"]},
                oracle=[chrome_open(url), sleep(3)] + star,
                near_misses=[("bookmarked_other", [chrome_open(f"{BASE}/{other}"), sleep(3)] + star),
                             ("opened_not_bookmarked", [chrome_open(url), sleep(5)])],
                extra_setup=site_setup(), settle_s=6, params={"pages": len(pages)})


# ------------------------------------------------------------------------------------------------ GIMP

def _img_ops(rng):
    from PIL import Image, ImageOps
    return {
        "rotate 90 degrees clockwise": lambda im: im.transpose(Image.Transpose.ROTATE_270),
        "rotate 90 degrees counter-clockwise": lambda im: im.transpose(Image.Transpose.ROTATE_90),
        "rotate 180 degrees": lambda im: im.transpose(Image.Transpose.ROTATE_180),
        "flip horizontally (mirror left-right)": lambda im: ImageOps.mirror(im),
        "flip vertically (upside down)": lambda im: ImageOps.flip(im),
        "convert to grayscale": lambda im: ImageOps.grayscale(im).convert("RGB"),
        "invert the colors": lambda im: ImageOps.invert(im.convert("RGB")),
    }


def image_transform(rng, idx):
    w, h = pick(rng, [(320, 200), (300, 220), (360, 240), (280, 210)])
    im = image(rng, w, h)
    ops = _img_ops(rng)
    op = pick(rng, sorted(ops))
    path = f"{HOME}/Pictures/{pick(rng, ['photo', 'chart', 'logo', 'scan', 'banner'])}_{rng.randint(1, 99):02d}.png"
    want = ops[op](im)
    others = [o for o in sorted(ops) if o != op and png_bytes(ops[o](im)) != png_bytes(want)]
    o1, o2 = rng.sample(others, 2)
    tol = 14.0 if "grayscale" in op else 4.0
    instr = phrase(rng, [
        "In GIMP, {op} the image {f} and save it back to the same file as PNG (overwrite it).",
        "Edit {f}: {op}. Export the result over the original file, PNG format.",
        "The picture {f} is open in GIMP. {Op} and overwrite {n} with the result."],
        op=op, Op=op[0].upper() + op[1:], f=path.replace(HOME, "~"), n=path.rsplit("/", 1)[1])
    return Spec(family="gimp_transform", app="gimp", instruction=instr, files={path: png_bytes(im)},
                check={"kind": "image", "path": path, "expected_b64": base64.b64encode(png_bytes(want)).decode(),
                       "format": "PNG", "max_mae": tol},
                oracle=write_file_steps(path, png_bytes(want)),
                near_misses=[(f"other_op:{o1}", write_file_steps(path, png_bytes(ops[o1](im)))),
                             (f"other_op:{o2}", write_file_steps(path, png_bytes(ops[o2](im))))],
                open_cmd=["gimp", "-n", path], allow=[path], settle_s=14, params={"op": op, "size": [w, h]})


def image_resize(rng, idx):
    from PIL import Image
    w, h = pick(rng, [(400, 300), (480, 320), (360, 270), (500, 250)])
    im = image(rng, w, h)
    f = pick(rng, [0.5, 0.25, 2.0, 0.75])
    nw, nh = int(w * f), int(h * f)
    path = f"{HOME}/Pictures/{pick(rng, ['photo', 'map', 'poster', 'icon'])}_{rng.randint(1, 99):02d}.png"
    crop = rng.random() < 0.4
    if crop:
        x0, y0 = rng.randrange(0, w // 3), rng.randrange(0, h // 3)
        nw, nh = rng.randrange(w // 3, w - x0), rng.randrange(h // 3, h - y0)
        want = im.crop((x0, y0, x0 + nw, y0 + nh))
        op = f"crop it to the {nw} x {nh} pixel region whose top-left corner is at x={x0}, y={y0}"
        nm1 = im.crop((x0 + 10, y0 + 10, x0 + nw + 10, y0 + nh + 10))
        nm2 = im.resize((nw, nh), Image.Resampling.BICUBIC)
    else:
        want = im.resize((nw, nh), Image.Resampling.BICUBIC)
        op = f"scale it to exactly {nw} x {nh} pixels"
        nm1 = im.resize((nw, nh + 2), Image.Resampling.BICUBIC)
        nm2 = im.crop((0, 0, min(w, nw), min(h, nh))).resize((nw, nh))
    instr = phrase(rng, [
        "Open {f} in GIMP, {op}, and save it over the same file as PNG.",
        "Using GIMP, {op} ({f}), then overwrite the original PNG.",
        "{Op} for the image {f}; keep the same file name and PNG format."],
        f=path.replace(HOME, "~"), op=op, Op=op[0].upper() + op[1:])
    return Spec(family="gimp_resize_crop", app="gimp", instruction=instr, files={path: png_bytes(im)},
                check={"kind": "image", "path": path, "expected_b64": base64.b64encode(png_bytes(want)).decode(),
                       "format": "PNG", "max_mae": 0.5 if crop else 10.0},
                oracle=write_file_steps(path, png_bytes(want)),
                near_misses=[("off_by_some_pixels", write_file_steps(path, png_bytes(nm1))),
                             ("wrong_method", write_file_steps(path, png_bytes(nm2)))],
                open_cmd=["gimp", "-n", path], allow=[path], settle_s=14, params={"crop": crop, "size": [w, h], "new": [nw, nh]})


def image_export(rng, idx):
    w, h = pick(rng, [(320, 200), (300, 220)])
    im = image(rng, w, h)
    stem = f"{pick(rng, ['photo', 'chart', 'logo', 'scan'])}_{rng.randint(1, 99):02d}"
    src = f"{HOME}/Pictures/{stem}.png"
    out = f"{HOME}/Pictures/{stem}.jpg"
    instr = phrase(rng, [
        "Export {f} as a JPEG image named {n} in the same folder (keep the PNG).",
        "Save a JPG copy of {f} as {o}.", "Convert the picture {f} to JPEG format, saved as {o}."],
        f=src.replace(HOME, "~"), n=f"{stem}.jpg", o=out.replace(HOME, "~"))
    from PIL import ImageOps
    return Spec(family="gimp_export_jpeg", app="gimp", instruction=instr, files={src: png_bytes(im)},
                check={"kind": "all", "checks": [
                    {"kind": "image", "path": out, "expected_b64": base64.b64encode(png_bytes(im)).decode(), "format": "JPEG", "max_mae": 6.0},
                    {"kind": "files_state", "must_exist": {src: sha256(png_bytes(im))}}]},
                oracle=write_file_steps(out, jpeg_bytes(im)),
                near_misses=[("png_renamed", write_file_steps(out, png_bytes(im))),
                             ("mirrored", write_file_steps(out, jpeg_bytes(ImageOps.mirror(im)))),
                             ("source_moved", [{"type": "execute", "command": ["rm", src], "shell": False}] + write_file_steps(out, jpeg_bytes(im)))],
                open_cmd=["gimp", "-n", src], allow=[out], settle_s=14, params={"size": [w, h]})


# ------------------------------------------------------------------------------------------------ VS Code

SETTINGS = f"{HOME}/.config/Code/User/settings.json"
BASE_SETTINGS = {"security.workspace.trust.enabled": False, "security.workspace.trust.startupPrompt": "never",
                 "security.workspace.trust.emptyWindow": False}
KNOBS = [("editor.fontSize", [12, 14, 16, 18, 20]), ("editor.tabSize", [2, 3, 4, 8]),
         ("files.autoSave", ["afterDelay", "onFocusChange", "onWindowChange"]),
         ("editor.wordWrap", ["on", "bounded", "wordWrapColumn"]), ("editor.lineNumbers", ["relative", "off"]),
         ("editor.minimap.enabled", [False]), ("files.trimTrailingWhitespace", [True]),
         ("editor.renderWhitespace", ["all", "boundary", "trailing"]), ("editor.cursorStyle", ["block", "underline"])]


def _settings(d):
    return (json.dumps(d, indent=4) + "\n").encode()


def vscode_setting(rng, idx):
    key, vals = pick(rng, KNOBS)
    val = pick(rng, vals)
    proj = f"{HOME}/projects/{pick(rng, ['api', 'site', 'tools', 'scripts'])}{rng.randint(1, 99)}"
    files = {f"{proj}/main.py": b"print('hello')\n"}
    want = dict(BASE_SETTINGS, **{key: val})
    wrong_val = next((v for v in (vals + [99, "x"]) if v != val and type(v) is type(val)), not val if isinstance(val, bool) else None)
    instr = phrase(rng, [
        "In VS Code, set the user setting {k} to {v} (User settings, not the workspace).",
        "Change VS Code's {k} user setting to {v}.", "Configure VS Code so that {k} is {v} for the user."],
        k=key, v=json.dumps(val))
    return Spec(family="vscode_setting", app="vscode", instruction=instr, files=files,
                check={"kind": "json_keys", "path": SETTINGS, "keys": {key: val}, "unchanged": BASE_SETTINGS},
                oracle=write_file_steps(SETTINGS, _settings(want)),
                near_misses=[("wrong_value", write_file_steps(SETTINGS, _settings(dict(BASE_SETTINGS, **{key: wrong_val})))),
                             ("workspace_not_user", write_file_steps(f"{proj}/.vscode/settings.json", _settings({key: val}))),
                             ("dropped_other_keys", write_file_steps(SETTINGS, _settings({key: val})))],
                open_cmd=["code", proj], allow=[f"{proj}/.vscode/*"], settle_s=8, params={"key": key, "value": val})


# function bank: (name, description, buggy, correct, near-miss versions, test generator)
def _fn_bank(rng):
    a, b = rng.randint(2, 9), rng.randint(10, 40)
    words = rng.sample(["apple", "Banana", "kiwi", "Cherry", "melon", "Grape", "lemon", "Olive"], 4)
    s = " ".join(words)
    return [
        ("count_vowels", "return the number of vowels (a, e, i, o, u, upper or lower case) in the string s",
         "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
         [("lower_only_kept", "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIO')\n"),
          ("counts_y", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiouy')\n")],
         [f"m.count_vowels({s!r}) == {sum(1 for c in s.lower() if c in 'aeiou')}", "m.count_vowels('AEIOU xyz') == 5",
          "m.count_vowels('') == 0", "m.count_vowels('rhythm') == 0"]),
        ("median", "return the median of the list xs (the mean of the two middle values for an even length)",
         "def median(xs):\n    return xs[len(xs) // 2]\n",
         "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
         [("no_even_case", "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n"),
          ("no_sort", "def median(xs):\n    n = len(xs)\n    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2\n")],
         [f"m.median([{b}, {a}, {a + b}]) == {sorted([a, b, a + b])[1]}", f"m.median([{a}, {b}, 1, 100]) == {(sorted([a, b, 1, 100])[1] + sorted([a, b, 1, 100])[2]) / 2}",
          "m.median([5]) == 5", "m.median([3, 1, 2]) == 2"]),
        ("clamp", "return x limited to the closed range [lo, hi]",
         "def clamp(x, lo, hi):\n    return max(x, hi)\n",
         "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
         [("upper_only", "def clamp(x, lo, hi):\n    return min(x, hi)\n"),
          ("swapped", "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n")],
         [f"m.clamp({b}, {a}, {a + 5}) == {a + 5}", f"m.clamp(-{a}, 0, {b}) == 0", f"m.clamp({a}, 0, {b}) == {a}", "m.clamp(5, 5, 5) == 5"]),
        ("dedupe", "return the items of xs without duplicates, keeping the first occurrence order",
         "def dedupe(xs):\n    return list(set(xs))\n",
         "def dedupe(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
         [("sorted", "def dedupe(xs):\n    return sorted(set(xs))\n"),
          ("last_occurrence", "def dedupe(xs):\n    return list(dict.fromkeys(reversed(xs)))[::-1]\n")],
         [f"m.dedupe([{b}, {a}, {b}, 1, {a}]) == [{b}, {a}, 1]", "m.dedupe([]) == []", "m.dedupe([3, 3, 3]) == [3]",
          "m.dedupe(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']"]),
        ("parse_minutes", "parse a duration like '1h30m', '45m' or '2h' and return the total minutes as an int",
         "def parse_minutes(s):\n    h, _, m = s.partition('h')\n    return int(h) * 60 + int(m.rstrip('m'))\n",
         "import re\n\ndef parse_minutes(s):\n    h = re.search(r'(\\d+)h', s)\n    m = re.search(r'(\\d+)m', s)\n    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)\n",
         [("hours_only", "import re\n\ndef parse_minutes(s):\n    h = re.search(r'(\\d+)h', s)\n    return int(h.group(1)) * 60 if h else 0\n"),
          ("minutes_as_hours", "import re\n\ndef parse_minutes(s):\n    h = re.search(r'(\\d+)h', s)\n    m = re.search(r'(\\d+)m', s)\n    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) * 60 if m else 0)\n")],
         [f"m.parse_minutes('{a}h{b}m') == {a * 60 + b}", f"m.parse_minutes('{b}m') == {b}", f"m.parse_minutes('{a}h') == {a * 60}",
          "m.parse_minutes('0h1m') == 1"]),
        ("word_lengths", "return a dict mapping each word of text (split on whitespace, lower-cased) to its length",
         "def word_lengths(text):\n    return {w: len(w) for w in text.split(',')}\n",
         "def word_lengths(text):\n    return {w.lower(): len(w) for w in text.split()}\n",
         [("not_lowered", "def word_lengths(text):\n    return {w: len(w) for w in text.split()}\n"),
          ("counts_instead", "def word_lengths(text):\n    out = {}\n    for w in text.lower().split():\n        out[w] = out.get(w, 0) + 1\n    return out\n")],
         [f"m.word_lengths({s!r}) == {({w.lower(): len(w) for w in s.split()})!r}", "m.word_lengths('') == {}",
          "m.word_lengths('Aa bbb') == {'aa': 2, 'bbb': 3}"]),
    ]


def code_fix(rng, idx):
    bank = _fn_bank(rng)
    name, desc, buggy, good, nms, tests = pick(rng, bank)
    proj = f"{HOME}/projects/{pick(rng, ['utils', 'lib', 'helpers', 'core'])}{rng.randint(1, 99)}"
    path = f"{proj}/{pick(rng, ['mathutil', 'textutil', 'funcs', 'helpers'])}.py"
    nonce = "T6-OK-" + "".join(rng.choice("abcdef0123456789") for _ in range(16))
    header = f'"""Utilities."""\n\n\n'
    instr = phrase(rng, [
        "The function {n} in {f} has a bug. Fix it so that it does this: {d}. Keep its name and arguments, and save the file.",
        "Open {f} in VS Code and correct {n}: it should {d2}. Save.",
        "{f} contains {n}, which should {d2}, but it is wrong. Fix the code and save it."],
        n=name, f=path.replace(HOME, "~"), d=desc, d2=desc.replace("return", "return", 1))
    return Spec(family="vscode_fix_function", app="vscode", instruction=instr, files={path: (header + buggy).encode()},
                check={"kind": "python_tests", "path": path, "tests": tests, "nonce": nonce, "function": name},
                oracle=write_file_steps(path, (header + good).encode()),
                near_misses=[(n2, write_file_steps(path, (header + code).encode())) for n2, code in nms],
                open_cmd=["code", proj, path], allow=[path, f"{proj}/__pycache__/*"], settle_s=8,
                params={"function": name})


def rename_symbol(rng, idx):
    old = pick(rng, ["calc", "proc", "helper", "do_it", "fn1", "work"]) + str(rng.randint(1, 9))
    new = pick(rng, ["compute_total", "process_items", "apply_discount", "scale_values", "sum_squares"])
    k = rng.randint(2, 6)
    proj = f"{HOME}/projects/{pick(rng, ['app', 'svc', 'job'])}{rng.randint(1, 99)}"
    path = f"{proj}/{pick(rng, ['main', 'logic', 'pipeline'])}.py"
    nonce = "T6-OK-" + "".join(rng.choice("abcdef0123456789") for _ in range(16))

    def src(defn, calls):
        return (f"def {defn}(xs):\n    return [x * {k} for x in xs]\n\n\n"
                f"def report(xs):\n    return sum({calls}(xs))\n\n\nRESULT = {calls}([1, 2, 3])\n").encode()
    tests = [f"m.{new}([1, 2]) == [{k}, {2 * k}]", f"m.report([1, 1]) == {2 * k}", f"not hasattr(m, '{old}')",
             f"m.RESULT == [{k}, {2 * k}, {3 * k}]"]
    instr = phrase(rng, [
        "In {f}, rename the function {o} to {n} everywhere it is defined or called. Save the file.",
        "Rename {o} to {n} in {f} (the definition and all uses). Save.",
        "Use VS Code to rename the function {o} as {n} throughout {f}, then save."],
        f=path.replace(HOME, "~"), o=old, n=new)
    return Spec(family="vscode_rename_symbol", app="vscode", instruction=instr, files={path: src(old, old)},
                check={"kind": "python_tests", "path": path, "tests": tests, "nonce": nonce, "function": new},
                oracle=write_file_steps(path, src(new, new)),
                near_misses=[("definition_only", write_file_steps(path, src(new, old))),
                             ("alias_kept", write_file_steps(path, src(new, new) + f"{old} = {new}\n".encode()))],
                open_cmd=["code", proj, path], allow=[path, f"{proj}/__pycache__/*"], settle_s=8,
                params={"old": old, "new": new})


FAMILIES = {"chrome_open_tabs": (open_tabs, 40), "chrome_download": (download, 40), "chrome_bookmark": (bookmark, 30),
            "gimp_transform": (image_transform, 60), "gimp_resize_crop": (image_resize, 40), "gimp_export_jpeg": (image_export, 30),
            "vscode_setting": (vscode_setting, 40), "vscode_fix_function": (code_fix, 50), "vscode_rename_symbol": (rename_symbol, 30)}
