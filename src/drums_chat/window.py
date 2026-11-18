"""The visual-context window of the drums-chat harness: which screenshots of the history go to the model, at what size.

This module is embedded verbatim into the chat program verifiers runs per rollout (drums_chat/__init__.py), so it uses
the standard library only; Pillow is imported only when a screenshot is downscaled (the program then declares it).

Only tool messages that carry an image are windowed; the system prompt and the first user message (the task text and the
first screenshot) are never touched, so they stay a byte-stable prefix of every request. Counting the n image-carrying
tool messages oldest first:

    newest `full` (screenshots_full)               sent as they are
    the next ones up to `total` (screenshots_total) downscaled by `downscale` (a factor of width and height), or sent as
                                                    they are when downscale is None or 1
    older ones                                      image removed, the text ("click(...): done.") kept

`stride` (screenshots_stride) moves the two boundaries in steps of `stride` messages instead of one. With stride 1 (the
0.2.2 rule) every new screenshot pushes one older screenshot across a boundary, so the request differs from the previous
one a few messages from its end and the inference server's prefix cache recomputes everything after that point on every
turn. With stride s the boundaries move on one turn in s; on the other turns the request is the previous request plus the
new messages, byte for byte, and the whole history is a cache hit. The price is up to s - 1 extra images in context.
"""
import base64 as _b64
import copy as _copy
import hashlib as _hashlib
import io as _io


def boundary(n: int, keep: int | None, stride: int = 1) -> int:
    """Index (into the n image messages, oldest first) of the first one kept: n - keep, rounded down to a multiple of
    stride, so the boundary moves only every `stride` messages. keep None keeps all."""
    if keep is None or n <= keep:
        return 0
    return ((n - keep) // max(1, stride)) * max(1, stride)


def image_messages(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "tool" and isinstance(m.get("content"), list)
            and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])]


_SMALL: dict = {}


def downscale_url(url: str, factor: float) -> str:
    """A data-URL image resized by `factor` (PNG, LANCZOS). Deterministic, and cached by content: the same screenshot
    becomes the same bytes on every turn, which the prefix cache needs."""
    key = (_hashlib.sha256(url.encode()).hexdigest(), factor)
    hit = _SMALL.get(key)
    if hit is not None:
        return hit
    from PIL import Image
    head, _, data = url.partition(",")
    im = Image.open(_io.BytesIO(_b64.b64decode(data))).convert("RGB")
    size = (max(1, int(round(im.width * factor))), max(1, int(round(im.height * factor))))
    buf = _io.BytesIO()
    im.resize(size, Image.Resampling.LANCZOS).save(buf, format="PNG")
    out = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode()
    if len(_SMALL) > 512:
        _SMALL.clear()
    _SMALL[key] = out
    return out


def _small_part(p, factor):
    if isinstance(p, dict) and p.get("type") == "image_url":
        url = (p.get("image_url") or {}).get("url") or ""
        if url.startswith("data:"):
            return {**p, "image_url": {**p["image_url"], "url": downscale_url(url, factor)}}
    return p


def window(messages: list, full: int | None, total: int | None = None, downscale: float | None = None,
           stride: int = 1) -> list:
    """The messages to send. The input list (the program's own history) is not changed."""
    if full is None:
        return messages
    total = full if total is None or total < full else total
    idx = image_messages(messages)
    n = len(idx)
    full_start = boundary(n, full, stride)
    keep_start = boundary(n, total, stride)
    shrink = downscale is not None and 0 < downscale < 1
    out = _copy.copy(messages)
    for j, i in enumerate(idx[:full_start]):
        m = dict(out[i])
        if j < keep_start:
            m["content"] = [p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "image_url")]
        elif shrink:
            m["content"] = [_small_part(p, downscale) for p in m["content"]]
        else:
            continue
        out[i] = m
    return out


def trim_images(messages: list, keep: int | None) -> list:
    """The 0.2.2 rule (newest `keep` screenshots, older ones removed): window(messages, keep)."""
    return window(messages, keep)
