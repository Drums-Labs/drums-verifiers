"""Computer-use actions as pyautogui code, and screenshots as model observations.

The model sees every screenshot resized to the observation size (default 1280x720) and gives coordinates in that space;
they are scaled to the desktop's real size (drumsd envs: 1920x1080) before they run. Each action is one `pyautogui` step
in the env, the same code path OSWorld's DesktopEnv uses for its pyautogui action space."""
from __future__ import annotations

import io
import json
import struct

KEY_ALIASES = {"return": "enter", "esc": "escape", "del": "delete", "cmd": "command", "super": "win", "pgup": "pageup",
               "pgdn": "pagedown", "control": "ctrl", "option": "alt"}
BUTTONS = ("left", "right", "middle")


class ActionError(ValueError):
    """The model asked for something that is not a valid action (told back to the model, not an env failure)."""


def png_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png[16:24])


def to_screen(x: float, y: float, obs: tuple[int, int], screen: tuple[int, int]) -> tuple[int, int]:
    """Observation pixel -> real screen pixel. Out-of-range coordinates are refused (not clamped)."""
    ow, oh = obs
    sw, sh = screen
    if not (0 <= x < ow and 0 <= y < oh):
        raise ActionError(f"coordinate ({x}, {y}) is outside the {ow}x{oh} screenshot")
    return int(round(x * sw / ow)), int(round(y * sh / oh))


def _keys(keys: str) -> list[str]:
    parts = [p.strip().lower() for p in str(keys).replace(" ", "+").split("+") if p.strip()]
    if not parts:
        raise ActionError("no key given")
    return [KEY_ALIASES.get(p, p) for p in parts]


def click(x, y, obs, screen, button="left", clicks=1) -> str:
    if button not in BUTTONS:
        raise ActionError(f"button must be one of {BUTTONS}")
    if int(clicks) not in (1, 2, 3):
        raise ActionError("clicks must be 1, 2 or 3")
    sx, sy = to_screen(x, y, obs, screen)
    return f"pyautogui.click({sx}, {sy}, clicks={int(clicks)}, button={button!r})"


def move(x, y, obs, screen) -> str:
    sx, sy = to_screen(x, y, obs, screen)
    return f"pyautogui.moveTo({sx}, {sy})"


def drag(x1, y1, x2, y2, obs, screen) -> str:
    a, b = to_screen(x1, y1, obs, screen), to_screen(x2, y2, obs, screen)
    return f"pyautogui.moveTo({a[0]}, {a[1]}); pyautogui.dragTo({b[0]}, {b[1]}, duration=0.5, button='left')"


def scroll(x, y, amount, obs, screen) -> str:
    sx, sy = to_screen(x, y, obs, screen)
    return f"pyautogui.moveTo({sx}, {sy}); pyautogui.scroll({int(amount)})"


def type_text(text) -> str:
    if not isinstance(text, str) or not text:
        raise ActionError("text must be a non-empty string")
    return f"pyautogui.write({json.dumps(text)}, interval=0.02)"


def key(keys) -> str:
    ks = _keys(keys)
    if len(ks) == 1:
        return f"pyautogui.press({ks[0]!r})"
    return "pyautogui.hotkey(" + ", ".join(repr(k) for k in ks) + ")"


def observation(png: bytes, obs: tuple[int, int]) -> bytes:
    """The screenshot resized to the observation size (PNG). A screenshot already at that size is returned as is."""
    if png_size(png) == tuple(obs):
        return png
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB").resize(tuple(obs), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def data_url(png: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(png).decode()
