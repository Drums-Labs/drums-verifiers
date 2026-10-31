"""The ScaleCUA OSWorld agent format: system prompt, the single `computer` tool, and tool call -> pyautogui.

Use it (task config `tool_style = "scalecua"`) when the policy is a ScaleCUA checkpoint, so the model sees the input it was
trained on: its system prompt, one `computer` tool with relative 0..1000 coordinates, the first screenshot with the
instruction, and each later screenshot as the tool result ("Success" + image, 1280x720).

PROVENANCE AND LICENCE. Copied from drums-lab experiments/exp3/env/scalecua_format.py, which is derived from
THUDM/SCALE-CUA (commit 3929e2fe364623153f2caa94ead71dc1aea50fb0) osworld_eval/mm_agents/scalecua-osworld/utils.py and
main.py, under the Apache License 2.0 (Copyright the SCALE-CUA authors). This file is distributed under the Apache
License 2.0 with this notice. The functions below are copies; one is MODIFIED (Apache-2.0 section 4(b)):
  * 2026-10-31, build_system_prompt: removed the line giving the model the image's sudo password ("If you need a password for
    sudo, the password of the computer is 'password'."). In a Drums env the desktop user has no privilege escalation, and an
    agent is never handed a credential.
"""
from datetime import datetime

DONE, FAIL = "DONE", "FAIL"
IMAGE_SIZE = (1280, 720)
RELATIVE_SPACE = 1000


ACTION_ENUM = ["key", "hold_key", "type", "cursor_position", "mouse_move", "left_mouse_down", "left_mouse_up",
               "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click",
               "scroll", "wait", "screenshot"]


def build_system_prompt(today=None):
    """SYSTEM_PROMPT from osworld_eval/mm_agents/scalecua-osworld/utils.py, with the date substituted the same way
    (`datetime.today().strftime('%A, %B %d, %Y')`). Text unchanged otherwise, except the removed sudo-password line (see the
    module docstring: a modification of the Apache-2.0 upstream)."""
    date = (today or datetime.today()).strftime('%A, %B %d, %Y')
    return f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY: You can declare a task infeasible at any point during execution - whether at the beginning after taking a screenshot, or later after attempting some actions and discovering barriers. Carefully evaluate whether the task is feasible given the current system state, available applications, and task requirements. If you determine that a task cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory or impossible requirements
  - Any other fundamental barriers that make completion impossible
  Then you MUST output exactly "[INFEASIBLE]" (including the square brackets) anywhere in your response to trigger the fail action. The system will automatically detect this pattern and terminate the task appropriately.
* The current date is {date}.
* Home directory of this Ubuntu system is '/home/user'.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""


def make_computer_tools(relative_coordinate=True, image_size=IMAGE_SIZE):
    """make_computer_tools from utils.py, unchanged: the single `computer` function tool."""
    resolution_str = f"{RELATIVE_SPACE}x{RELATIVE_SPACE}" if relative_coordinate else f"{image_size[0]}x{image_size[1]}"
    tool_description = (
        "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
        "* This is an interface to a desktop GUI. You do not have direct access to a "
        "terminal (you may have access to a terminal application) or applications menu. "
        "You must click on desktop icons to start applications.\n"
        "* Some applications may take time to start or process actions, so you may need "
        "to wait to see the results of your actions. E.g. if you click on Firefox and a "
        "window doesn't open, try waiting before taking another screenshot.\n"
        f"* The screen's resolution is {resolution_str}.\n"
        "* You should ONLY call mouse_move if you intend to hover over an element "
        "without clicking. Otherwise, use the click or drag functions directly.\n"
        "* Whenever you intend to click on an element like an icon, you should consult "
        "a screenshot to determine the coordinates of the element before moving the cursor.\n"
        "* If you tried clicking on a program or link but it failed to load, even after "
        "waiting, try adjusting your click location so that the tip of the cursor visually "
        "falls on the element that you want to click.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in "
        "the center of the element. Don't click boxes on their edges unless asked."
    )
    return [{
        "type": "function",
        "function": {
            "name": "computer",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "description": "The action to perform. The available actions are:\n* `key`: Press a key or key-combination on the keyboard.\n  - This supports xdotool's `key` syntax.\n  - Examples: \"a\", \"Return\", \"alt+Tab\", \"ctrl+s\", \"Up\", \"KP_0\" (for the numpad 0 key).\n* `hold_key`: Hold down a key or multiple keys for a specified duration (in seconds). Supports the same syntax as `key`.\n* `type`: Type a string of text on the keyboard.\n* `cursor_position`: Get the current (x, y) pixel coordinate of the cursor on the screen.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_mouse_down`: Press the left mouse button.\n* `left_mouse_up`: Release the left mouse button.\n* `left_click`: Click the left mouse button at the specified (x, y) pixel coordinate on the screen. You can also include a key combination to hold down while clicking using the `text` parameter.\n* `left_click_drag`: Click and drag the cursor from `start_coordinate` to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button at the specified (x, y) pixel coordinate on the screen.\n* `middle_click`: Click the middle mouse button at the specified (x, y) pixel coordinate on the screen.\n* `double_click`: Double-click the left mouse button at the specified (x, y) pixel coordinate on the screen.\n* `triple_click`: Triple-click the left mouse button at the specified (x, y) pixel coordinate on the screen.\n* `scroll`: Scroll the screen in a specified direction by a specified amount of clicks of the scroll wheel, at the specified (x, y) pixel coordinate. DO NOT use PageUp/PageDown to scroll.\n* `wait`: Wait for a specified duration (in seconds).\n* `screenshot`: Take a screenshot of the screen.",
                        "enum": list(ACTION_ENUM),
                        "type": "string"
                    },
                    "coordinate": {
                        "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "duration": {
                        "description": "The duration to hold the key down for. Required only by `action=hold_key` and `action=wait`.",
                        "type": "integer"
                    },
                    "scroll_amount": {
                        "description": "The number of 'clicks' to scroll. Required only by `action=scroll`.",
                        "type": "integer"
                    },
                    "scroll_direction": {
                        "description": "The direction to scroll the screen. Required only by `action=scroll`.",
                        "enum": ["up", "down", "left", "right"],
                        "type": "string"
                    },
                    "start_coordinate": {
                        "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to start the drag from. Required only by `action=left_click_drag`.",
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "text": {
                        "description": "Required only by `action=type`, `action=key`, and `action=hold_key`. Can also be used by click or scroll actions to hold down keys while clicking or scrolling.",
                        "type": "string"
                    }
                },
                "required": ["action"]
            }
        }
    }]


KEY_CONVERSION = {"page_down": "pagedown", "page_up": "pageup", "super_l": "win", "super": "command", "escape": "esc"}


def _scale(coord, screen_w, screen_h, relative=True):
    """resize_factor of the eval agent: relative -> screen/1000; absolute -> screen/1280, screen/720. int() truncates."""
    if relative:
        fx, fy = screen_w / RELATIVE_SPACE, screen_h / RELATIVE_SPACE
    else:
        fx, fy = screen_w / IMAGE_SIZE[0], screen_h / IMAGE_SIZE[1]
    return (int(coord[0] * fx), int(coord[1] * fy))


def args_to_pyautogui(args, screen_w, screen_h, relative_coordinate=True):
    """parse_actions_from_tool_call from main.py, as a function of the parsed arguments. Returns the pyautogui code string
    (several lines), or the sentinel "DONE" / "FAIL" / "CALL_USER" for the legacy terminal branches. Raises ValueError on an
    invalid action or argument shape (the eval agent lets that propagate to its caller)."""
    result = ""
    action = args.get("action")
    action = {"left click": "click", "right click": "right_click"}.get(action, action)
    text = args.get("text")
    coordinate = args.get("coordinate")
    start_coordinate = args.get("start_coordinate")
    scroll_direction = args.get("scroll_direction")
    scroll_amount = args.get("scroll_amount")
    duration = args.get("duration")
    if coordinate:
        coordinate = _scale(coordinate, screen_w, screen_h, relative_coordinate)
    if start_coordinate:
        start_coordinate = _scale(start_coordinate, screen_w, screen_h, relative_coordinate)

    if action == "left_mouse_down":
        result += "pyautogui.mouseDown()\n"
    elif action == "left_mouse_up":
        result += "pyautogui.mouseUp()\n"
    elif action == "hold_key":
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        for key in text.split('+'):
            result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ValueError(f"coordinate is required for {action}")
        if text is not None:
            raise ValueError(f"text is not accepted for {action}")
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{coordinate} must be a tuple of length 2")
        if not all(isinstance(i, int) for i in coordinate):
            raise ValueError(f"{coordinate} must be a tuple of ints")
        x, y = coordinate
        if action == "mouse_move":
            result += f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
        else:
            if start_coordinate:
                if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                    raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                if not all(isinstance(i, int) for i in start_coordinate):
                    raise ValueError(f"{start_coordinate} must be a tuple of ints")
                sx, sy = start_coordinate
                result += f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})\n"
            result += f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"
    elif action in ("key", "type"):
        if text is None:
            raise ValueError(f"text is required for {action}")
        if coordinate is not None:
            raise ValueError(f"coordinate is not accepted for {action}")
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        if action == "key":
            keys = text.split('+')
            for key in keys:
                key = KEY_CONVERSION.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyDown('{key}')\n"
            for key in reversed(keys):
                key = KEY_CONVERSION.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyUp('{key}')\n"
        else:
            for char in text:
                if char == '\n':
                    result += "pyautogui.press('enter')\n"
                elif char == "'":
                    result += 'pyautogui.press("\'")\n'
                elif char == '\\':
                    result += "pyautogui.press('\\\\')\n"
                elif char == '"':
                    result += "pyautogui.press('\"')\n"
                else:
                    result += f"pyautogui.press('{char}')\n"
    elif action == "scroll":
        if text is not None:
            result += f"pyautogui.keyDown('{text.lower()}')\n"
        if coordinate is None:
            if scroll_direction in ("up", "down"):
                result += f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})\n"
            elif scroll_direction in ("left", "right"):
                result += f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})\n"
        else:
            x, y = coordinate
            if scroll_direction in ("up", "down"):
                result += f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})\n"
            elif scroll_direction in ("left", "right"):
                result += f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})\n"
        if text is not None:
            result += f"pyautogui.keyUp('{text.lower()}')\n"
    elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
        if text:
            for key in text.split('+'):
                result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
        fn = {"left_click": "click", "right_click": "rightClick", "double_click": "doubleClick",
              "middle_click": "middleClick", "triple_click": "tripleClick"}.get(action)
        if coordinate is not None:
            x, y = coordinate
            if action == "left_press":
                result += f"pyautogui.mouseDown({x}, {y})\ntime.sleep(1)\npyautogui.mouseUp({x}, {y})\n"
            else:
                result += f"pyautogui.{fn}({x}, {y})\n"
        else:
            if action == "left_press":
                result += "pyautogui.mouseDown()\ntime.sleep(1)\npyautogui.mouseUp()\n"
            else:
                result += f"pyautogui.{fn}()\n"
        if text:
            for key in reversed(text.split('+')):
                result += f"pyautogui.keyUp('{key.strip().lower()}')\n"
    elif action == "wait":
        result += "pyautogui.sleep(0.5)\n"
    elif action == "fail":
        result += FAIL
    elif action == "done":
        result += DONE
    elif action == "call_user":
        result += "CALL_USER"
    elif action == "screenshot":
        result += "pyautogui.sleep(0.1)\n"
    else:
        raise ValueError(f"Invalid action: {action}")
    return result
