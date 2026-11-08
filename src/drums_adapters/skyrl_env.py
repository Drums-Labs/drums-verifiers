"""SkyRL-Gym environment: one Drums desktop per episode (skyrl_gym BaseTextEnv).

Register it in your SkyRL entrypoint and point the dataset's env_class at it:

    from skyrl_gym.envs import register
    register(id="drums", entry_point="drums_adapters.skyrl_env:DrumsEnv")

Each dataset row's `extra_info` (SkyRL passes it as `extras`) carries the Drums task JSON under "drums_task" (a task
from the factory's tasks.jsonl, or any OSWorld-format task). The policy acts with hybrid tool calls
(drums_adapters.common); the episode ends on DONE, a turn with no tool call, or max_turns.

Reward: the checker's score when the reward is trusted (valid episode, no checker error, no audit flag). SkyRL has no
per-sample drop, so an untrusted reward is returned as 0.0 with metadata["drums_untrusted"] = True and the audit:
filter those samples (e.g. in a custom generator) instead of training on them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .common import Episode, connect

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = connect()
    return _CLIENT


class DrumsEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: Dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        task = extras.get("drums_task") or (extras.get("extra_info") or {}).get("drums_task")
        if isinstance(task, str):
            task = json.loads(task)
        if not isinstance(task, dict):
            raise ValueError("DrumsEnv needs extras['drums_task'] (a Drums task JSON)")
        self.task = task
        cfg = env_config or {}
        self.max_turns = int((cfg.get("max_turns") if hasattr(cfg, "get") else None) or extras.get("max_turns") or 15)
        self.ep = None

    def init(self, prompt) -> Tuple[Any, Dict[str, Any]]:
        self.ep = Episode(_client(), self.task, self.max_turns)
        msgs = list(prompt or [])
        if not msgs:
            msgs = [{"role": "user", "content": self.ep.prompt()}]
        return msgs, {"drums_env": self.ep.env.id, "task_id": self.task.get("id")}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        obs, done = self.ep.step(action)
        if not done:
            return BaseTextEnvStepOutput(observations=[{"role": "user", "content": obs}], reward=0.0, done=False,
                                         metadata={"tool_result": obs[:200]})
        r = self.ep.finish()
        meta = {"drums": {k: r[k] for k in ("score", "valid", "checker_error", "audit_flags", "trusted", "env", "episode", "turns")},
                "drums_untrusted": r["reward"] is None}
        return BaseTextEnvStepOutput(observations=[], reward=float(r["reward"] or 0.0), done=True, metadata=meta)

    def close(self):
        if self.ep is not None:
            self.ep.close()
            self.ep = None
