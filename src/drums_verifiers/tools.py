"""The computer: one MCP tool server per rollout, bound to the rollout's Drums env (TaskData.drums_env_id).

Every tool is one action in the env (a pyautogui step through the Drums SDK), then a pause, then a screenshot that comes back
to the model as an image (resized to the observation size). Coordinates are pixels of that screenshot.

An infrastructure failure (drumsd down, the env gone, a guest error) is not the model's fault: the tool records it in the
rollout state (`infra_error`), the task's stop hook ends the rollout before the next model call, and the reward refuses to
score it, so the trace ends as an error that prime-rl leaves out of training. The model only ever sees "the environment
failed", never a made-up observation.

Run as a module: `python -m drums_verifiers.tools` (verifiers launches it per rollout)."""

import asyncio
import functools
import inspect
import json
import time
from typing import Annotated, Literal

import verifiers.v1 as vf
from mcp.server.mcpserver import Image
from pydantic import Field

from . import actions as A
from . import scalecua as S
from . import sdk


class DrumsState(vf.State):
    """Per-rollout state shared by the tool server and the task (synced on every tool call)."""
    actions: int = 0
    invalid_actions: int = 0
    infra_error: str | None = None
    quarantined: str | None = None
    """drumsd quarantined the env mid-episode (a gate failed, e.g. a restored process exited within 5 s): the episode is
    invalid, not broken; the task drops it."""
    action_s: float = 0.0
    transcript: list[dict] = []
    """One entry per tool call, in order (when the env's `transcript` flag is on): tool, arguments (JSON, cut to
    ARGS_MAX characters), status, latency_s (the whole call: action, pause and screenshot), t (unix time at the end)."""


ARGS_MAX = 200
"""Characters of a tool call's JSON arguments kept in the transcript."""


def call_status(before: tuple, after: tuple, result) -> str:
    """What a tool call did, from the state counters around it and its result: ok, invalid (the model's arguments were
    refused), env_failed, quarantined, skipped (the env had stopped earlier; nothing ran) or no_action."""
    (inv0, infra0, quar0), (inv1, infra1, quar1) = before, after
    if quar1 and not quar0:
        return "quarantined"
    if infra1 and not infra0:
        return "env_failed"
    if inv1 > inv0:
        return "invalid"
    if isinstance(result, str) and result.startswith("The environment stopped earlier"):
        return "skipped"
    if isinstance(result, str) and result.endswith(": no action."):
        return "no_action"
    return "ok"


class ComputerConfig(vf.ToolsetConfig):
    style: Literal["drums", "scalecua", "hybrid"] = "drums"
    """drums: one tool per action, pixel coordinates. scalecua: the ScaleCUA agent's single `computer` tool, prompt and
    observation format (for ScaleCUA checkpoints). hybrid: the drums GUI tools plus shell, file and app-API tools in the
    same desktop (drums.hybrid): a file the browser downloads is visible to the shell tool at once."""
    obs_width: int = 1280
    obs_height: int = 720
    pause_s: float = 1.0
    """Seconds between an action and its screenshot (the desktop settles)."""
    url: str | None = None
    """drumsd URL; default $DRUMS_URL, then http://127.0.0.1:7070."""
    token_file: str | None = None
    """drumsd token file; default $DRUMS_TOKEN_FILE, then /etc/drums/token, then $DRUMS_TOKEN."""
    relative_coordinates: bool = True
    """scalecua style only: coordinates in 0..1000 (the ScaleCUA RL checkpoints' setting); False = screenshot pixels."""


class DesktopToolset(vf.Toolset[ComputerConfig, DrumsState]):
    """The env binding and the act -> wait -> screenshot step shared by both tool styles."""

    def _with_state(self, fn):
        """Every tool call is also appended to the rollout's transcript (state.transcript), inside the state sync."""
        name = getattr(fn, "tool_name", None) or fn.__name__
        tool = f"{self.server_name}_{name}" if self.server_name else name

        @functools.wraps(fn)
        async def logged(*args, **kwargs):
            if not getattr(self, "transcript_on", True):
                return await fn(*args, **kwargs)
            st = self.state
            before = (st.invalid_actions, st.infra_error, st.quarantined)
            t0, status, result = time.monotonic(), "error", None
            try:
                result = await fn(*args, **kwargs)
                st = self.state
                status = call_status(before, (st.invalid_actions, st.infra_error, st.quarantined), result)
                return result
            finally:
                try:
                    a = json.dumps(kwargs, default=str, ensure_ascii=False)
                except (TypeError, ValueError):
                    a = str(kwargs)
                self.state.transcript.append({"tool": tool, "args": a[:ARGS_MAX], "status": status,
                                              "latency_s": round(time.monotonic() - t0, 3), "t": round(time.time(), 3)})
        logged.__signature__ = inspect.signature(fn)
        return super()._with_state(logged)

    async def setup_task(self, task) -> None:
        self.transcript_on = bool(getattr(task, "drums_transcript", True))
        self.env_id = getattr(task, "drums_env_id", None)
        self.env = None
        self.screen = None
        if not self.env_id:
            return

        def bind():
            env = sdk.env(self.env_id, self.config.url, self.config.token_file)
            return env, A.png_size(env.screenshot())
        try:
            self.env, self.screen = await asyncio.to_thread(bind)
        except Exception as e:  # noqa: BLE001 - surfaced through the first tool call's infra_error
            self.bind_error = f"{type(e).__name__}: {e}"[:500]

    @property
    def obs(self):
        return (self.config.obs_width, self.config.obs_height)

    async def _do(self, describe: str, code: str | None, ok_text: str | None = None):
        """Run one action (or none, for screenshot/wait), then return [text, screenshot]."""
        st = self.state
        if st.infra_error or st.quarantined:
            return f"The environment stopped earlier ({st.infra_error or st.quarantined}); no action was taken."
        if self.env is None:
            st.infra_error = getattr(self, "bind_error", None) or "no Drums env bound to this rollout"
            return f"The environment is unavailable ({st.infra_error})."

        def run():
            t0 = time.monotonic()
            if code is not None:
                r = self.env.step({"type": "pyautogui", "code": code})
                res = r.get("result") or {}
                if isinstance(res, dict) and res.get("returncode") not in (None, 0):
                    raise RuntimeError(f"action exited {res.get('returncode')}: {str(res.get('error'))[:300]}")
            if self.config.pause_s:
                time.sleep(self.config.pause_s)
            png = self.env.screenshot()
            return png, time.monotonic() - t0
        try:
            png, dt = await asyncio.to_thread(run)
        except Exception as e:  # noqa: BLE001 - every env failure ends the rollout (dropped if quarantined, else an error)
            rec = {}
            try:
                rec = await asyncio.to_thread(self.env.refresh)
            except Exception:  # noqa: BLE001
                pass
            if rec.get("state") == "quarantined":
                st.quarantined = f"{describe}: quarantined: {rec.get('reason') or rec.get('last_error') or e}"[:500]
            else:
                st.infra_error = f"{describe}: {type(e).__name__}: {e}"[:500]
            return f"The environment stopped while running {describe}; the episode is over."
        st.action_s += dt
        if code is not None:
            st.actions += 1
        return [ok_text or f"{describe}: done.", Image(data=A.observation(png, self.obs), format="png")]

    def _invalid(self, e: Exception):
        self.state.invalid_actions += 1
        return f"Invalid action: {e}"


class ComputerToolset(DesktopToolset):
    """Drums' own tools: one tool per action, coordinates in screenshot pixels."""
    TOOL_PREFIX = "computer"

    @vf.tool
    async def screenshot(self):
        """Take a screenshot of the desktop."""
        return await self._do("screenshot", None)

    @vf.tool
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """Click at (x, y) in screenshot pixels. button: left, right or middle; clicks: 1, 2 (double) or 3."""
        try:
            code = A.click(x, y, self.obs, self.screen or self.obs, button, clicks)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"click({x}, {y}, {button}, {clicks})", code)

    @vf.tool
    async def move(self, x: int, y: int):
        """Move the mouse to (x, y) in screenshot pixels."""
        try:
            code = A.move(x, y, self.obs, self.screen or self.obs)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"move({x}, {y})", code)

    @vf.tool
    async def drag(self, x1: int, y1: int, x2: int, y2: int):
        """Drag with the left button from (x1, y1) to (x2, y2)."""
        try:
            code = A.drag(x1, y1, x2, y2, self.obs, self.screen or self.obs)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"drag({x1}, {y1} -> {x2}, {y2})", code)

    @vf.tool
    async def scroll(self, x: int, y: int, amount: int):
        """Scroll at (x, y): positive amount scrolls up, negative scrolls down."""
        try:
            code = A.scroll(x, y, amount, self.obs, self.screen or self.obs)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"scroll({x}, {y}, {amount})", code)

    @vf.tool
    async def type_text(self, text: str):
        """Type text with the keyboard."""
        try:
            code = A.type_text(text)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"type_text({text!r})", code)

    @vf.tool
    async def key(self, keys: str):
        """Press a key or a combination, e.g. "enter", "f2", "ctrl+a", "ctrl+shift+t"."""
        try:
            code = A.key(keys)
        except A.ActionError as e:
            return self._invalid(e)
        return await self._do(f"key({keys!r})", code)

    @vf.tool
    async def wait(self, seconds: float = 1.0):
        """Wait up to 10 seconds, then take a screenshot."""
        s = max(0.0, min(float(seconds), 10.0))
        await asyncio.sleep(s)
        return await self._do(f"wait({s})", None)


class HybridToolset(ComputerToolset):
    """GUI + shell + files + app APIs (drums.hybrid) on the rollout's one desktop. Every non-GUI tool returns text (JSON);
    an infrastructure failure ends the rollout exactly as a GUI action's does."""

    async def _hy(self, describe: str, name: str, args: dict):
        st = self.state
        if st.infra_error or st.quarantined:
            return f"The environment stopped earlier ({st.infra_error or st.quarantined}); no action was taken."
        if self.env is None:
            st.infra_error = getattr(self, "bind_error", None) or "no Drums env bound to this rollout"
            return f"The environment is unavailable ({st.infra_error})."
        from drums.hybrid import HybridTools
        import json as _json

        def run():
            t0 = time.monotonic()
            out = HybridTools(self.env).call(name, args)
            return out, time.monotonic() - t0
        try:
            out, dt = await asyncio.to_thread(run)
        except Exception as e:  # noqa: BLE001
            st.infra_error = f"{describe}: {type(e).__name__}: {e}"[:500]
            return f"The environment stopped while running {describe}; the episode is over."
        st.action_s += dt
        st.actions += 1
        return _json.dumps(out)[:20000]

    @vf.tool
    async def shell(self, command: str, timeout_s: float = 60):
        """Run a bash command in the desktop as the desktop user (no root, no internet); returns returncode, stdout, stderr."""
        return await self._hy(f"shell({command!r})", "shell", {"command": command, "timeout_s": timeout_s})

    @vf.tool
    async def read_file(self, path: str):
        """Read a file in the desktop (~ is /home/user): text as text, binary as base64."""
        return await self._hy(f"read_file({path!r})", "read_file", {"path": path})

    @vf.tool
    async def write_file(self, path: str, content: str):
        """Write a text file in the desktop (creates folders, overwrites)."""
        return await self._hy(f"write_file({path!r})", "write_file", {"path": path, "content": content})

    @vf.tool
    async def list_dir(self, path: str):
        """List a folder in the desktop."""
        return await self._hy(f"list_dir({path!r})", "list_dir", {"path": path})

    @vf.tool
    async def office_convert(self, path: str, format: str, outdir: str | None = None):  # noqa: A002
        """Convert a document with LibreOffice (docx->pdf, xlsx->csv, csv->xlsx, ...)."""
        a = {"path": path, "format": format}
        if outdir:
            a["outdir"] = outdir
        return await self._hy(f"office_convert({path!r}, {format!r})", "office_convert", a)

    @vf.tool
    async def spreadsheet_read(self, path: str):
        """Read the first sheet of a spreadsheet as rows of text (through LibreOffice)."""
        return await self._hy(f"spreadsheet_read({path!r})", "spreadsheet_read", {"path": path})

    @vf.tool
    async def browser_tabs(self):
        """List the open Chrome tabs."""
        return await self._hy("browser_tabs()", "browser_tabs", {})

    @vf.tool
    async def browser_open(self, url: str):
        """Open a URL in a new Chrome tab."""
        return await self._hy(f"browser_open({url!r})", "browser_open", {"url": url})


_P = S.make_computer_tools(relative_coordinate=True)[0]["function"]["parameters"]["properties"]


def _d(name):
    return _P[name]["description"]


class ScaleCuaToolset(DesktopToolset):
    """The ScaleCUA agent's single `computer` tool (Anthropic computer-use vocabulary), its argument names, enums and
    descriptions, and its action -> pyautogui mapping (scalecua.py, unchanged). The tool result is "Success" + screenshot."""
    TOOL_PREFIX = None

    def register(self, mcp) -> None:
        desc = S.make_computer_tools(self.config.relative_coordinates, self.obs)[0]["function"]["description"]
        mcp.add_tool(self._with_state(self.computer), name="computer", description=desc)

    async def computer(
        self,
        action: Annotated[Literal[tuple(S.ACTION_ENUM)], Field(description=_d("action"))],
        coordinate: Annotated[list[int] | None, Field(description=_d("coordinate"))] = None,
        duration: Annotated[int | None, Field(description=_d("duration"))] = None,
        scroll_amount: Annotated[int | None, Field(description=_d("scroll_amount"))] = None,
        scroll_direction: Annotated[Literal["up", "down", "left", "right"] | None, Field(description=_d("scroll_direction"))] = None,
        start_coordinate: Annotated[list[int] | None, Field(description=_d("start_coordinate"))] = None,
        text: Annotated[str | None, Field(description=_d("text"))] = None,
    ):
        args = {k: v for k, v in dict(action=action, coordinate=coordinate, duration=duration, scroll_amount=scroll_amount,
                                       scroll_direction=scroll_direction, start_coordinate=start_coordinate, text=text).items()
                if v is not None}
        sw, sh = self.screen or self.obs
        try:
            code = S.args_to_pyautogui(args, sw, sh, self.config.relative_coordinates)
        except (ValueError, TypeError) as e:
            return self._invalid(e)
        if code in (S.DONE, S.FAIL, "CALL_USER"):
            return f"{code}: no action."
        return await self._do(f"computer({args})", code, ok_text="Success")


if __name__ == "__main__":
    import json
    import os
    # verifiers launches `python -m drums_verifiers.tools` for both styles; the class comes from the config it passes
    style = json.loads(os.environ.get("VF_CONFIG") or "{}").get("style", "drums")
    {"scalecua": ScaleCuaToolset, "hybrid": HybridToolset}.get(style, ComputerToolset).run()
