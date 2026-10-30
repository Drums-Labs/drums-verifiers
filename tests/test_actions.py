import io

import pytest
from PIL import Image

from drums_verifiers import actions as A

OBS, SCREEN = (1280, 720), (1920, 1080)


def png(size):
    b = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(b, format="PNG")
    return b.getvalue()


def test_coordinates_scale_from_the_screenshot_to_the_screen():
    assert A.to_screen(1235, 590, OBS, SCREEN) == (1852, 885)      # the e0df folder icon
    assert A.click(0, 0, OBS, SCREEN) == "pyautogui.click(0, 0, clicks=1, button='left')"
    with pytest.raises(A.ActionError):
        A.to_screen(1280, 10, OBS, SCREEN)                          # outside the screenshot: refused, not clamped
    with pytest.raises(A.ActionError):
        A.click(10, 10, OBS, SCREEN, button="side")


def test_keys_and_text_are_quoted_safely():
    assert A.key("ctrl+a") == "pyautogui.hotkey('ctrl', 'a')"
    assert A.key("Return") == "pyautogui.press('enter')"
    code = A.type_text("a'b\"c\n); import os")
    assert code.startswith("pyautogui.write(") and compile(code, "x", "exec")
    with pytest.raises(A.ActionError):
        A.type_text("")


def test_observation_is_resized_to_the_observation_size():
    out = A.observation(png(SCREEN), OBS)
    assert A.png_size(out) == OBS
    same = png(OBS)
    assert A.observation(same, OBS) is same
