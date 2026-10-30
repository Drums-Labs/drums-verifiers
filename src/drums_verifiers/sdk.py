"""One place that connects to drumsd. The URL and token never go into task data or traces: the URL comes from config or
$DRUMS_URL, the token from a token file ($DRUMS_TOKEN_FILE, default /etc/drums/token) or $DRUMS_TOKEN."""
from __future__ import annotations

import functools
import os

DEFAULT_URL = "http://127.0.0.1:7070"
DEFAULT_TOKEN_FILE = "/etc/drums/token"


def resolve(url: str | None = None, token_file: str | None = None) -> tuple[str, str | None]:
    url = url or os.environ.get("DRUMS_URL") or DEFAULT_URL
    path = token_file or os.environ.get("DRUMS_TOKEN_FILE")
    if path is None and not os.environ.get("DRUMS_TOKEN") and os.path.exists(DEFAULT_TOKEN_FILE):
        path = DEFAULT_TOKEN_FILE
    if path:
        with open(path) as f:
            return url, f.read().strip()
    return url, os.environ.get("DRUMS_TOKEN")


@functools.lru_cache(maxsize=8)
def client(url: str | None = None, token_file: str | None = None):
    import drums
    u, tok = resolve(url, token_file)
    return drums.connect(u, tok)


def env(env_id: str, url: str | None = None, token_file: str | None = None):
    """An existing env by id (the rollout's env, created by DrumsEnv)."""
    import drums
    c = client(url, token_file)
    return drums.Env(c, c.call("GET", f"/v1/slots/{env_id}"))
