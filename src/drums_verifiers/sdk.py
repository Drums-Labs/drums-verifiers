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


def no_sdk_admission_retry():
    """The SDK's 429 retry policy turned off, or None for an SDK without one (< 0.4.0).

    drums-sdk 0.4.0 retries 429 Overloaded inside the client, for up to 60 s by default. DrumsEnv already waits for
    admission itself (DrumsEnv._admitted: bounded by admission_wait_s, every wait recorded in the episode's
    admission_waits). Both at once would stretch each refused call by the SDK's budget and hide the waits from the
    record, so this package keeps admission in one place: its own loop, with the SDK's retry off (max_wait_s=0, the
    0.3.0 behaviour)."""
    import drums
    policy = getattr(drums, "RetryPolicy", None)
    return policy(max_wait_s=0) if policy is not None else None


@functools.lru_cache(maxsize=8)
def client(url: str | None = None, token_file: str | None = None):
    import drums
    u, tok = resolve(url, token_file)
    retry = no_sdk_admission_retry()
    return drums.connect(u, tok, retry=retry) if retry is not None else drums.connect(u, tok)


def env(env_id: str, url: str | None = None, token_file: str | None = None):
    """An existing env by id (the rollout's env, created by DrumsEnv)."""
    import drums
    c = client(url, token_file)
    return drums.Env(c, c.call("GET", f"/v1/slots/{env_id}"))
