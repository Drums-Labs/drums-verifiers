"""verl multi-turn tool: one Drums desktop per rollout (verl.tools.base_tool.BaseTool).

verl's ToolAgentLoop (and its sglang/vLLM multi-turn rollout) calls create() when a rollout starts, execute() for every
tool call the model makes, calc_reward() at the end and release() when it is done. Configure it in the tools YAML:

    tools:
      - class_name: drums_adapters.verl_tool.DrumsDesktopTool
        config: {type: native, max_turns: 15}
        tool_schema: null        # the schema below is used

and pass the Drums task in the dataset row's tools_kwargs: {"drums_desktop": {"create_kwargs": {"drums_task": {...}}}}.

One tool, `drums_desktop`, takes {"name": <a drums.hybrid tool>, "arguments": {...}}: GUI, shell, files and app APIs on
the same desktop. Screenshots come back as images in the ToolResponse.

Reward: calc_reward() returns the checker's score when the reward is trusted (valid episode, no checker error, no audit
flag), else 0.0; the reason and the audit are kept in the instance's record (`last_reward`), and `trusted` tells a
reward manager to mask the sample.
"""
from __future__ import annotations

import io
import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .common import Episode, connect

SCHEMA = {"type": "function", "function": {
    "name": "drums_desktop",
    "description": "Act on the Ubuntu desktop. name is one of: gui (pyautogui code), screenshot, shell, read_file, write_file, "
                   "list_dir, office_convert, spreadsheet_read, browser_tabs, browser_open; arguments are that tool's.",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string", "description": "the desktop tool"},
        "arguments": {"type": "object", "description": "the tool's arguments"}}, "required": ["name"]}}}


class DrumsDesktopTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema or OpenAIFunctionToolSchema(**SCHEMA))
        self._eps: dict[str, Episode] = {}
        self._tasks: dict[str, dict] = {}
        self._owned: set = set()
        self._client = None
        self.records: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Records the task only: the desktop is made on the first call (standalone use) or is the trajectory's
        (DrumsToolAgentLoop, which verl's per-call create/release cannot disturb)."""
        instance_id = instance_id or str(uuid4())
        task = kwargs.get("drums_task") or (kwargs.get("create_kwargs") or {}).get("drums_task")
        if isinstance(task, str):
            task = json.loads(task)
        self._tasks[instance_id] = task
        return instance_id, ToolResponse(text=(task or {}).get("instruction", ""))

    async def _own_episode(self, instance_id):
        import asyncio
        if instance_id not in self._eps:
            if self._client is None:
                self._client = connect()
            self._eps[instance_id] = await asyncio.to_thread(Episode, self._client, self._tasks[instance_id],
                                                             int(self.config.get("max_turns", 15)))
            self._owned.add(instance_id)
        return self._eps[instance_id]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        import asyncio
        ep = await self._own_episode(instance_id)
        name, args = parameters.get("name"), parameters.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        try:
            out = await asyncio.to_thread(ep.tools.call, name, args)
        except Exception as e:  # noqa: BLE001 - the model sees its tool error
            return ToolResponse(text=f"tool error: {type(e).__name__}: {e}"[:500]), 0.0, {"error": True}
        ep.turns += 1
        ep.calls.append(name)
        if isinstance(out, (bytes, bytearray)):
            if not self.config.get("images", True):     # a text-only model (no processor): verl refuses image data
                return ToolResponse(text=f"[screenshot taken: {len(out)} bytes PNG; images are off for this model]"), 0.0, {}
            from PIL import Image
            return ToolResponse(image=[Image.open(io.BytesIO(out)).convert("RGB")], text="screenshot"), 0.0, {}
        return ToolResponse(text=json.dumps(out)[:4000]), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        import asyncio
        r = await asyncio.to_thread((await self._own_episode(instance_id)).finish)
        self.records[instance_id] = r
        return float(r["reward"] or 0.0)

    async def release(self, instance_id: str, **kwargs) -> None:
        ep = self._eps.pop(instance_id, None)
        self._tasks.pop(instance_id, None)
        if ep is not None and instance_id in self._owned:
            ep.close()
        self._owned.discard(instance_id)


# ------------------------------------------------------------------------------------------------ the agent loop
# verl's ToolAgentLoop creates and releases a tool instance around every single call, so a desktop cannot live in a
# tool instance. It lives in the trajectory instead: DrumsToolAgentLoop opens one Episode per trajectory before the
# loop starts (from the row's extra_info / tools_kwargs "drums_task"), the tool finds it through agent_data, and the
# loop scores it when the trajectory ends (reward_score, plus the audit in extra_fields).

_EPISODES: dict[str, Episode] = {}


def _episode_for(agent_data) -> Episode | None:
    key = (getattr(agent_data, "extra_fields", None) or {}).get("drums_episode")
    return _EPISODES.get(key) if key else None


try:
    from verl.experimental.agent_loop.agent_loop import register
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    @register("drums_tool_agent")
    class DrumsToolAgentLoop(ToolAgentLoop):
        """ToolAgentLoop with one Drums desktop per trajectory and the Drums reward (score + audit) at the end."""

        async def run(self, sampling_params, priority: int = 0, **kwargs):
            import asyncio
            extra = kwargs.get("extra_info") or {}
            task = extra.get("drums_task") or ((kwargs.get("tools_kwargs") or {}).get("drums_desktop") or {}).get(
                "create_kwargs", {}).get("drums_task")
            if isinstance(task, str):
                task = json.loads(task)
            ep = await asyncio.to_thread(Episode, connect(), task, int(extra.get("max_turns", 15)))
            key = uuid4().hex
            _EPISODES[key] = ep
            _PENDING.set(key)
            try:
                out = await super().run(sampling_params, priority=priority, **kwargs)
                r = await asyncio.to_thread(ep.finish)
            finally:
                _EPISODES.pop(key, None)
                await asyncio.to_thread(ep.close)
            out.reward_score = r["reward"] if r["reward"] is not None else 0.0
            out.extra_fields.update({"drums_reward": r["reward"], "drums_trusted": r["trusted"], "drums_valid": r["valid"],
                                     "drums_score": r["score"], "drums_audit_flags": r["audit_flags"], "drums_env": r["env"],
                                     "drums_tools_used": r["tools_used"]})
            return out
except ImportError:  # pragma: no cover - verl without the experimental agent loop
    DrumsToolAgentLoop = None

import contextvars  # noqa: E402

_PENDING: contextvars.ContextVar = contextvars.ContextVar("drums_pending_episode", default=None)


async def _execute_in_trajectory(self, instance_id, parameters, **kwargs):
    """DrumsDesktopTool.execute when called from DrumsToolAgentLoop: the trajectory's episode, not a per-call one."""
    agent_data = kwargs.get("agent_data")
    if agent_data is not None and "drums_episode" not in agent_data.extra_fields and _PENDING.get():
        agent_data.extra_fields["drums_episode"] = _PENDING.get()
    ep = _episode_for(agent_data) if agent_data is not None else None
    if ep is not None:
        self._eps[instance_id] = ep
    try:
        return await _execute_plain(self, instance_id, parameters, **kwargs)
    finally:
        if ep is not None:
            self._eps.pop(instance_id, None)       # the loop owns the episode; release() must not close it


_execute_plain = DrumsDesktopTool.execute
DrumsDesktopTool.execute = _execute_in_trajectory
