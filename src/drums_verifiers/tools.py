"""The computer: one MCP tool server per rollout, bound to the rollout's Drums env (TaskData.drums_env_id).

Every tool is one action in the env (a pyautogui step through the Drums SDK), then a pause, then a screenshot that comes back
to the model as an image (resized to the observation size). Coordinates are pixels of that screenshot.

An infrastructure failure (drumsd down, the env gone, a guest error) is not the model's fault: the tool records it in the
rollout state (`infra_error`), the task's stop hook ends the rollout before the next model call, and the reward refuses to
score it, so the trace ends as an error that prime-rl leaves out of training. The model only ever sees "the environment
failed", never a made-up observation.

Run as a module: `python -m drums_verifiers.tools` (verifiers launches it per rollout)."""

import asyncio
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


class ComputerConfig(vf.ToolsetConfig):
    style: Literal["drums", "scalecua"] = "drums"
    """drums: one tool per action, pixel coordinates. scalecua: the ScaleCUA agent's single `computer` tool, prompt and
    observation format (for ScaleCUA checkpoints)."""
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

    async def setup_task(self, task) -> None:
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
    (ScaleCuaToolset if style == "scalecua" else ComputerToolset).run()
