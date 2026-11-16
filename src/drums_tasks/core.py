"""Task factory core: seeded generation, deterministic files, setup steps and scripted solutions.

A generated task has two halves:

- the task JSON drumsd runs (OSWorld format: `id`, `instruction`, `config` setup steps, and a `drums_verifier`
  evaluator that names a function in `drums_t6.py` with JSON options, a redundant checker and a scope manifest);
- its solutions, kept apart from the task (an agent never sees them): the scripted oracle that must score 1, at least
  two near-miss solutions that must score 0, and a scope probe (the oracle plus one edit outside the task's scope)
  that the reward audit must flag.

Everything is a function of (family, index, seed): the same seed rebuilds byte-identical tasks (office files are
rewritten with fixed zip timestamps and fixed document properties).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
import zipfile

HOME = "/home/user"
VERIFIER = "drums_t6"
CHUNK = 60_000          # base64 characters per argv string in SETUP steps (Linux MAX_ARG_STRLEN is 128 KiB)
# drumsd validates every string of an action's argv at <= 8192 characters (`invalid step body: action.command.N Too
# big`); ACTION steps (oracle, near misses) chunk below that. Setup steps travel in the task config, not a step body.
ACTION_CHUNK = 8_000
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

# the soffice used by the oracles: a private profile, so it never touches the running GUI instance's profile
SOFFICE = ("soffice", "-env:UserInstallation=file:///tmp/t6-oracle-lo", "--headless", "--norestore")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rng_for(seed: int, family: str, index: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{family}:{index}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def normalize_zip(data: bytes) -> bytes:
    """Rewrite a zip (xlsx/docx/pptx) with fixed timestamps and permissions, same member order: deterministic bytes."""
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            zi = zipfile.ZipInfo(info.filename, FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            body = src.read(info.filename)
            if info.filename == "docProps/core.xml":
                # writers stamp "now" into the modified/created dates at save time
                body = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)", rb"\g<1>2026-01-01T00:00:00Z\g<2>", body)
            dst.writestr(zi, body)
    return out.getvalue()


# ------------------------------------------------------------------------------------------------ actions
# Actions are drumsd `env.step` actions. `write_file` / `remove` / `sh` build execute actions that run as the desktop
# user, the same rights an agent's shell tool has.

_WRITE = "import base64,os,sys;p=sys.argv[1];os.makedirs(os.path.dirname(p),exist_ok=True);" \
         "open(p,sys.argv[2]).write(base64.b64decode(sys.argv[3]))"


def write_file_steps(path: str, data: bytes, kind: str = "action") -> list:
    """Execute steps that write `data` to `path` in the desktop (chunked under the argv limit). kind: action | setup."""
    b64 = base64.b64encode(data).decode()
    size = ACTION_CHUNK if kind == "action" else CHUNK
    chunks = [b64[i:i + size] for i in range(0, len(b64), size)] or [""]
    out = []
    for i, c in enumerate(chunks):
        cmd = ["python3", "-c", _WRITE, path, "wb" if i == 0 else "ab", c]
        out.append({"type": "execute", "command": cmd, "shell": False} if kind == "action"
                   else {"type": "execute", "parameters": {"command": cmd, "shell": False}})
    return out


def sh(cmd: str) -> dict:
    return {"type": "execute", "command": ["bash", "-c", cmd], "shell": False}


def py(code: str, *args: str) -> dict:
    return {"type": "execute", "command": ["python3", "-c", code, *args], "shell": False}


def gui(code: str) -> dict:
    return {"type": "pyautogui", "code": code}


def sleep(s: float) -> dict:
    return {"type": "sleep", "seconds": s}


def lo_convert(src: str, fmt: str, outdir: str) -> dict:
    """LibreOffice (headless, private profile) converts src into outdir: the oracle's office files are written by
    LibreOffice itself, the same writer the agent's GUI saves go through."""
    return {"type": "execute", "command": [*SOFFICE, "--convert-to", fmt, "--outdir", outdir, src], "shell": False}


def office_write(path: str, data: bytes, fmt: str) -> list:
    """Write an office file the way the app would: stage `data` in /tmp, convert it with LibreOffice into the target
    folder under the target name, remove the staging copy."""
    import posixpath
    d, name = posixpath.split(path)
    stem, ext = posixpath.splitext(name)
    stage_dir = f"/tmp/t6-stage-{sha256(path.encode())[:8]}"
    stage = f"{stage_dir}/{stem}{ext}"
    out = write_file_steps(stage, data)
    # convert into a staging folder, then copy over the target: LibreOffice refuses to store over a file another
    # instance (the GUI the task opened) holds a lock on, and reports success anyway
    out.append({"type": "execute", "command": [*SOFFICE, "--convert-to", fmt, "--outdir", f"{stage_dir}/out", stage], "shell": False})
    out.append({"type": "execute", "command": ["cp", "-f", f"{stage_dir}/out/{stem}.{fmt}", path], "shell": False})
    out.append({"type": "execute", "command": ["rm", "-rf", stage_dir], "shell": False})
    return out


# ------------------------------------------------------------------------------------------------ setup steps

def setup_write(path: str, data: bytes) -> list:
    return write_file_steps(path, data, kind="setup")


def setup_exec(cmd: list) -> dict:
    return {"type": "execute", "parameters": {"command": cmd, "shell": False}}


def setup_launch(cmd: list) -> dict:
    return {"type": "launch", "parameters": {"command": cmd, "shell": False}}


def setup_sleep(s: float) -> dict:
    return {"type": "sleep", "parameters": {"seconds": s}}


# ------------------------------------------------------------------------------------------------ the task record

class Spec:
    """What a family's generator returns for one index. The builder turns it into a task and its solutions."""

    def __init__(self, *, family, app, instruction, files, check, oracle, near_misses, open_cmd=None, extra_setup=None,
                 redundant=True, allow=(), roots=None, protected=None, params=None, settle_s=None, scope_probe=None,
                 multi_app=False, difficulty=None):
        self.family, self.app, self.instruction = family, app, instruction
        self.files = dict(files)                   # absolute desktop path -> bytes, seeded at setup
        self.check = check                         # options for drums_t6.check (has "kind")
        self.oracle = oracle                       # list of actions
        self.near_misses = near_misses             # list of (name, actions)
        self.open_cmd = open_cmd                   # launched after the files are written
        self.extra_setup = extra_setup or []
        self.redundant = redundant
        self.allow = list(allow)                   # globs that may change
        self.roots = roots
        self.protected = protected                 # path the scope probe edits (default: a decoy file)
        self.params = params or {}
        self.settle_s = settle_s
        self.scope_probe = scope_probe
        self.multi_app = multi_app
        self.difficulty = difficulty


DEFAULT_ROOTS = [f"{HOME}/Desktop", f"{HOME}/Documents", f"{HOME}/Downloads", f"{HOME}/Pictures", f"{HOME}/projects"]
DECOY = f"{HOME}/Documents/README-keep.txt"


def build_task(spec: Spec, task_id: str, seed: int) -> tuple:
    """-> (task JSON for drumsd, solutions record)."""
    files = dict(spec.files)
    # every task has one decoy the instruction never mentions: the scope probe edits it, the audit must notice
    decoy_text = f"Notes for {task_id}. Keep this file unchanged.\n".encode()
    files.setdefault(DECOY, decoy_text)
    config = []
    for p in sorted(files):
        config += setup_write(p, files[p])
    config += spec.extra_setup
    if spec.open_cmd:
        config.append(setup_launch(spec.open_cmd))
        config.append(setup_sleep(spec.settle_s if spec.settle_s is not None else 6))
    roots = spec.roots or DEFAULT_ROOTS
    manifest = {p: sha256(b) for p, b in sorted(files.items()) if any(p.startswith(r + "/") for r in roots)}
    evaluator = {"func": "drums_verifier", "verifier": VERIFIER, "function": "check", "options": spec.check,
                 "deadline_s": 60,
                 "scope": {"roots": roots, "manifest": manifest, "allow": spec.allow}}
    if spec.redundant:
        evaluator["redundant"] = {"verifier": VERIFIER, "function": "check_alt"}
    task = {"id": task_id, "snapshot": spec.app, "instruction": spec.instruction, "source": "drums-t6-factory",
            "config": config, "related_apps": [spec.app] + (["multi"] if spec.multi_app else []), "evaluator": evaluator}
    protected = spec.protected or DECOY
    probe = spec.scope_probe or (list(spec.oracle) + [py(
        "import sys;open(sys.argv[1],'a').write('edited outside the task\\n')", protected)])
    sol = {"id": task_id, "family": spec.family, "app": spec.app, "seed": seed, "params": spec.params,
           "oracle": spec.oracle, "near_misses": [{"name": n, "actions": a} for n, a in spec.near_misses],
           "scope_probe": probe, "difficulty_hint": spec.difficulty}
    return task, sol


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def phrase(rng: random.Random, templates: list, **kw) -> str:
    return pick(rng, templates).format(**kw)
