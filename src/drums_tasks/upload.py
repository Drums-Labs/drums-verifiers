"""Uploaded verifiers, named by hash (drumsd docs/design/verifier-sandbox.md).

A task can name its lab verifier by the sha256 of its bytes instead of by a file the host's operator installed:

    "evaluator": {"func": "drums_verifier", "verifier": "drums_t6", "verifier_sha256": "<64 hex>", "function": "check", ...}

drumsd then runs it in its verifier sandbox, and refuses to create the env (422 `verifier_missing`) when it does not
have that hash. `ensure_verifiers` makes sure it does: for every hash a task set names, ask the host and upload the
local source if it is missing. Standard library only; works with any client that has `.url`, `.token` and
`.call(method, path)` (the drums SDK's Client).
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = {"drums_t6": os.path.join(HERE, "verifier", "drums_t6.py")}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def bundled_sources() -> dict:
    """{sha256: (name, bytes)} of the verifiers this package ships."""
    out = {}
    for name, p in BUNDLED.items():
        with open(p, "rb") as f:
            b = f.read()
        out[sha256_bytes(b)] = (name, b)
    return out


def sources_from_files(paths) -> dict:
    """{sha256: (name, bytes)} of local verifier files (the name is the file's stem)."""
    out = {}
    for p in paths or []:
        with open(p, "rb") as f:
            b = f.read()
        out[sha256_bytes(b)] = (os.path.splitext(os.path.basename(p))[0], b)
    return out


def task_hashes(task: dict) -> list:
    """The verifier hashes a task's evaluator names (the primary and its redundant checker), in order, no repeats."""
    ev = task.get("evaluator") or {}
    if ev.get("func") != "drums_verifier":
        return []
    out = []
    for h in (ev.get("verifier_sha256"), (ev.get("redundant") or {}).get("verifier_sha256")):
        if h and h not in out:
            out.append(h)
    return out


def name_by_hash(task: dict, sources: dict) -> dict:
    """A copy of `task` whose drums_verifier evaluator (and redundant checker) also names its verifier's sha256, found
    by the verifier's name in `sources` ({sha256: (name, bytes)}). The name stays, for display and for hosts that
    still score by file."""
    by_name = {n: h for h, (n, _) in sources.items()}
    t = copy.deepcopy(task)
    ev = t.get("evaluator") or {}
    if ev.get("func") != "drums_verifier":
        return t
    for part in (ev, ev.get("redundant")):
        if isinstance(part, dict):
            name = part.get("verifier") or ev.get("verifier")
            if name in by_name and not part.get("verifier_sha256"):
                part["verifier_sha256"] = by_name[name]
    return t


def _check_transport(url: str, client) -> None:
    u = urllib.parse.urlsplit(url)
    if u.scheme == "https" or getattr(client, "allow_insecure_http", False):
        return
    host = (u.hostname or "").strip("[]")
    try:
        loop = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loop = host == "localhost"
    if not loop:
        raise RuntimeError(f"refusing to send the Drums token over plain http to {host}: use https, or allow_insecure_http")


def upload(client, data: bytes, name: str | None = None, timeout: float = 60.0) -> dict:
    """POST /v1/verifiers with the source bytes; the stored record ({sha256, bytes, name, created, ...})."""
    url = client.url.rstrip("/") + "/v1/verifiers" + (f"?name={urllib.parse.quote(name)}" if name else "")
    _check_transport(url, client)
    headers = {"Content-Type": "text/x-python"}
    if getattr(client, "token", None):
        headers["Authorization"] = f"Bearer {client.token}"
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"verifier upload failed: HTTP {e.code}: {e.read()[:400].decode(errors='replace')}") from None


def ensure_verifiers(client, tasks, sources: dict | None = None, cache: dict | None = None) -> dict:
    """Make sure the host has every verifier `tasks` name by hash. {sha256: "present" | "uploaded"}. A hash with no
    local source raises LookupError naming it (pass the file). `cache` (a dict kept by the caller) skips hashes already
    checked on this client."""
    sources = {**bundled_sources(), **(sources or {})}
    cache = {} if cache is None else cache
    out = {}
    for t in tasks:
        for h in task_hashes(t):
            if h in out:
                continue
            if cache.get(h):
                out[h] = cache[h]
                continue
            try:
                client.call("GET", f"/v1/verifiers/{h}")
                out[h] = "present"
            except Exception as e:  # noqa: BLE001 - the SDK's NotFound, or any client's 404
                if getattr(e, "status", None) != 404:
                    raise
                if h not in sources:
                    raise LookupError(f"task {t.get('id')} names verifier {h}, which the host does not have and no local "
                                      "file hashes to (pass the verifier file)") from None
                name, data = sources[h]
                rec = upload(client, data, name)
                if rec.get("sha256") != h:
                    raise RuntimeError(f"the host stored verifier {name} as {rec.get('sha256')}, not {h}")
                out[h] = "uploaded"
            cache[h] = out[h]
    return out
