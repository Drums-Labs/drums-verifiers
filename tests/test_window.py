"""The drums-chat visual-context window: the default is the 0.2.2 rule, the throughput options downscale and drop older
screenshots, and the stride layout keeps the request a byte-stable prefix of the next one on most turns (what vLLM's
prefix cache needs). Offline; the requests are built the way verifiers' chat program builds them."""
import base64
import io
import json
import random

import pytest
from PIL import Image

from drums_chat import DrumsChatConfig, program_source
from drums_chat.window import boundary, trim_images, window

SYSTEM = {"role": "system", "content": "You control an Ubuntu desktop through tools. ..."}


def png_url(seed: int, size=(128, 72)) -> str:
    rnd = random.Random(seed)
    im = Image.new("RGB", size, (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    im.putpixel((seed % size[0], 0), (0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def first_user():
    return {"role": "user", "content": [{"type": "text", "text": "Task: copy the files"},
                                        {"type": "image_url", "image_url": {"url": png_url(0)}}]}


def turn(i: int) -> list:
    call = {"id": f"call_{i}", "type": "function", "function": {"name": "computer_click", "arguments": json.dumps({"x": i, "y": i})}}
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"call_{i}", "name": "computer_click",
             "content": [{"type": "text", "text": f"click({i}, {i}): done."}, {"type": "image_url", "image_url": {"url": png_url(i)}}]}]


def history(turns: int) -> list:
    out = [SYSTEM, first_user()]
    for i in range(1, turns + 1):
        out += turn(i)
    return out


def wire(messages) -> list[str]:
    return [json.dumps(m, sort_keys=True) for m in messages]


def requests(turns: int, **kw) -> list[list[str]]:
    """The request the model sees before each of `turns` model calls."""
    return [wire(window(history(t), **kw)) for t in range(turns)]


def shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def images(msgs) -> list[tuple[int, int]]:
    out = []
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image_url":
                    im = Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].partition(",")[2])))
                    out.append(im.size)
    return out


def legacy_trim(messages, keep):
    """drums_chat 0.2.2's trim_images, copied, to pin the default to it."""
    import copy
    if keep is None:
        return messages
    out = copy.copy(messages)
    idx = [i for i, m in enumerate(out) if m.get("role") == "tool" and isinstance(m.get("content"), list)
           and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])]
    for i in idx[:-keep] if keep > 0 else idx:
        m = dict(out[i])
        m["content"] = [p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "image_url")]
        out[i] = m
    return out


@pytest.mark.parametrize("keep", [None, 0, 1, 5, 20])
def test_default_is_the_0_2_2_rule(keep):
    for t in range(0, 16):
        h = history(t)
        assert wire(window(h, keep)) == wire(legacy_trim(h, keep)) == wire(trim_images(h, keep))
    cfg = DrumsChatConfig()
    assert cfg.window_args() == (5, 5, 0.5, 1) and not cfg.needs_pillow()
    assert "pillow" not in program_source(cfg.needs_pillow()).split("# ///", 2)[1]


def test_the_program_history_is_never_changed():
    h = history(10)
    before = wire(h)
    window(h, 2, 5, 0.5, 3)
    assert wire(h) == before


def test_full_downscaled_and_dropped_tiers():
    out = window(history(10), full=2, total=5, downscale=0.5)
    assert images(out) == [(64, 36)] * 3 + [(128, 72)] * 2          # 5 kept: 3 downscaled, the newest 2 full size
    tool_texts = [m["content"][0]["text"] for m in out if m["role"] == "tool"]
    assert len(tool_texts) == 10                                     # every tool message keeps its text
    assert out[1] == first_user()                                    # the task and the first screenshot are untouched
    assert images(window(history(10), full=2, total=5, downscale=None)) == [(128, 72)] * 5


def test_downscaled_bytes_are_the_same_every_turn():
    a = wire(window(history(9), full=2, total=5, downscale=0.5))
    b = wire(window(history(9), full=2, total=5, downscale=0.5))
    assert a == b


def test_system_prompt_and_task_are_a_stable_prefix_of_every_request():
    for kw in ({"full": 5}, {"full": 2, "total": 5, "downscale": 0.5}, {"full": 3, "total": 6, "downscale": 0.5, "stride": 3}):
        reqs = requests(15, **kw)
        assert all(r[:2] == reqs[0][:2] for r in reqs), kw


def test_sliding_window_breaks_the_prefix_on_every_turn_past_the_window():
    """Why D1's prefix-cache hit rate was 64%: with the 0.2.2 rule each turn removes one older screenshot, so the request
    differs from the previous one inside the history on every turn once more than 5 screenshots exist."""
    reqs = requests(15, full=5)
    breaks = [t for t in range(1, 15) if shared_prefix(reqs[t - 1], reqs[t]) < len(reqs[t - 1])]
    assert breaks == list(range(6, 15))      # request t holds t screenshots; the window first slides at the 6th one


@pytest.mark.parametrize("full,total,stride", [(3, 6, 3), (2, 5, 3), (4, 4, 4), (5, 5, 2)])
def test_stride_layout_is_byte_identical_up_to_the_new_content(full, total, stride):
    """With stride s the request before turn t + 1 is the request before turn t plus the new assistant and tool messages,
    byte for byte, except on one turn in s (where the window moves). On those turns the system prompt and task are still
    shared."""
    turns = 30
    reqs = requests(turns, full=full, total=total, downscale=0.5, stride=stride)
    breaks = []
    for t in range(1, turns):
        prev, cur = reqs[t - 1], reqs[t]
        if cur[:len(prev)] == prev:
            assert len(cur) == len(prev) + 2                        # only the new assistant + tool message were added
            assert "".join(cur).startswith("".join(prev))           # and the serialized request extends byte for byte
        else:
            breaks.append(t)
            assert shared_prefix(prev, cur) >= 2
    assert breaks, "the window must move at some point"
    assert all(b - a >= stride for a, b in zip(breaks, breaks[1:])), breaks
    assert len(breaks) <= (turns - total) // stride + 1
    for t in range(turns):                                          # never fewer than `full` full-size screenshots
        sizes = images(window(history(t), full, total, 0.5, stride))
        assert sizes.count((128, 72)) >= min(t, full) and len(sizes) < total + stride


def test_boundary():
    assert [boundary(n, 5, 1) for n in range(4, 9)] == [0, 0, 1, 2, 3]
    assert [boundary(n, 3, 3) for n in range(3, 10)] == [0, 0, 0, 3, 3, 3, 6]
    assert boundary(9, None, 3) == 0


def test_config_checks_and_program_declares_pillow_only_when_downscaling():
    with pytest.raises(ValueError):
        DrumsChatConfig(screenshots_full=4, screenshots_total=2)
    cfg = DrumsChatConfig(screenshots_full=3, screenshots_total=6, downscale=0.5, screenshots_stride=3)
    assert cfg.window_args() == (3, 6, 0.5, 3) and cfg.needs_pillow()
    src = program_source(True)
    assert '"pillow"' in src.split("# ///", 2)[1]
    compile(src, "drums_chat_program", "exec")
    assert not DrumsChatConfig(screenshots_full=3, screenshots_total=6, downscale=1.0).needs_pillow()
