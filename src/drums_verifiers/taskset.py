"""Drums computer-use tasks as a verifiers v1 taskset + env.

One rollout = one Drums env (an isolated desktop from drumsd) running one OSWorld task:
  DrumsEnv.run          creates the env with the task (drumsd runs the task's own setup), waits for the episode's gates,
                        and opens the conversation with the instruction and the first screenshot
  the agent             any harness with MCP tools (default: verifiers' `null` chat loop); the actions are the
                        `computer_*` tools (tools.py), each answered with a screenshot
  DrumsTask reward      env.verify(): OSWorld's own checker, run by drumsd, together with the episode's gate verdict
  the env is closed after scoring.

Invalid episodes never reach the trainer as a normal reward:
  - drumsd's verdict is not "valid" (a gate failed: isolation, mounts, network, cgroup limits, state hash, processes alive after
    the restore)  -> no reward is recorded and the trace is DROPPED (`trace.dropped = "drums_invalid:<gates>"`); prime-rl's
    algorithms skip dropped traces (iter_trainable_traces) and eval leaves them out of reward means.
  - the checker could not run, or the env failed mid-episode (tools.py infra_error) -> the reward raises, the trace ends as an
    ERROR (`trace.ok = False`); prime-rl skips errored traces too.
  The flagged score is kept in `trace.info["drums"]` for audit.

Group rollouts from one reached state (GRPO from a fork): with `fork_n = N`, one episode creates the parent env, plays the
`prefix` actions, lets the desktop settle, forks N children with env.fork(N) and runs one agent per child; each child is
scored by its own checker and verdict. The N traces of the episode are the group: prime-rl's `grpo` with `group_size = 1`
(the group is the one episode) or `hierarchical_grpo` with `episode_agents = ["agent"]`.

Every scored trace is also appended to a JSONL record (`record_path`): env ids, fork parent, verdict, failed gates, score,
reward handed out, drop reason, timings."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Iterator

from pydantic import Field

import verifiers.v1 as vf

from . import actions as A
from . import sdk
from .tools import ComputerConfig, ComputerToolset, DrumsState

SYSTEM_PROMPT = """You control an Ubuntu desktop through tools. Each tool performs one action and returns a screenshot.
Coordinates are pixels of the screenshot you see ({w}x{h}); (0, 0) is the top-left corner.
Tools: computer_click, computer_move, computer_drag, computer_scroll, computer_type_text, computer_key, computer_wait,
computer_screenshot. When the task is complete, reply with a short message and no tool call."""


class DrumsTaskData(vf.TaskData):
    osworld: dict
    """The OSWorld task JSON (id, instruction, config, evaluator), as drumsd serves it from its pinned OSWorld checkout."""
    drums_image: str = "osworld-ubuntu"
    drums_env_id: str | None = None
    """Set by DrumsEnv on the copy of the task each agent runs (the rollout's env); never set in the taskset."""
    drums_episode: int | None = None
    drums_steps: int | None = None
    """The env's lifetime step count when DrumsEnv handed it to the agent (the reuse guard compares it)."""
    drums_parent: str | None = None


class DrumsTaskConfig(vf.TaskConfig):
    tools: ComputerConfig = ComputerConfig()
    verify_timeout_s: float = 600.0


def step_count(env) -> int:
    """Lifetime number of steps drumsd ran on this env (its per-op metrics)."""
    return int((((env.metrics().get("usage") or {}).get("ops") or {}).get("step") or {}).get("n") or 0)


class DrumsTask(vf.Task[DrumsTaskData, DrumsState, DrumsTaskConfig]):
    @property
    def key(self) -> str:
        return f"osworld:{self.data.osworld.get('id')}"

    @classmethod
    def toolsets(cls, config: DrumsTaskConfig) -> list[vf.Toolset]:
        return [ComputerToolset(config.tools)]

    async def setup(self, trace: vf.Trace) -> None:
        """Refuse to start an agent on an env that is not exactly the one DrumsEnv prepared: same env, same episode, ready.
        (An agent-level retry would otherwise continue on a desktop a failed attempt already changed.)"""
        if not self.data.drums_env_id:
            raise RuntimeError("no Drums env: run this taskset with its DrumsEnv (the package's exported env)")
        cfg = self.config.tools

        def check():
            env = sdk.env(self.data.drums_env_id, cfg.url, cfg.token_file)
            rec = env.record
            if rec.get("state") != "ready" or rec.get("episode") != self.data.drums_episode:
                raise RuntimeError(f"env {env.id} is {rec.get('state')} at episode {rec.get('episode')}, "
                                   f"prepared at episode {self.data.drums_episode}")
            steps = step_count(env)
            if steps != self.data.drums_steps:
                raise RuntimeError(f"env {env.id} took {steps - (self.data.drums_steps or 0)} actions since it was prepared "
                                   "(an earlier attempt used it); not reusing it")
        await asyncio.to_thread(check)

    @vf.stop
    def env_failed(self, trace: vf.Trace) -> bool:
        """End the rollout before the next model call once the env failed (tools.py records it)."""
        return bool(trace.state.infra_error)

    @vf.reward
    async def task_success(self, trace: vf.Trace) -> float | None:
        if trace.state.infra_error:
            raise RuntimeError(f"Drums env failed during the episode: {trace.state.infra_error}")
        cfg = self.config.tools

        def verify():
            env = sdk.env(self.data.drums_env_id, cfg.url, cfg.token_file)
            t0 = time.monotonic()
            r = env.verify(strict=False)
            return r, time.monotonic() - t0
        r, dt = await asyncio.wait_for(asyncio.to_thread(verify), self.config.verify_timeout_s)
        info = {"env": r.env_id, "episode": r.episode, "task": r.task_id, "valid": r.valid, "verdict": r.verdict,
                "failed_gates": r.bad_gates, "score": r.score, "checker_error": r.checker_error, "verify_s": round(dt, 3),
                "actions": trace.state.actions, "invalid_actions": trace.state.invalid_actions, "parent": self.data.drums_parent}
        trace.info["drums"] = info
        trace.record_metric("drums_valid", float(r.valid))
        trace.record_metric("drums_actions", float(trace.state.actions))
        if r.checker_error is not None:
            raise RuntimeError(f"OSWorld checker failed on env {r.env_id}: {r.checker_error}")
        if not r.valid:
            trace.drop(f"drums_invalid:{r.verdict}:{','.join(map(str, r.bad_gates)) or '-'}")
            return None                        # no reward at all: never a normal 0
        return float(r.score)


class DrumsTasksetConfig(vf.TasksetConfig):
    task_ids: list[str] = Field(default_factory=lambda: ["e0df059f-28a6-4169-924f-b9623e7184cc"])
    """OSWorld task ids (or unique prefixes >= 8 chars), fetched from drumsd's pinned OSWorld checkout."""
    tasks_dir: str | None = None
    """Read the task JSONs from this directory (<id>.json) instead of drumsd."""
    image: str = "osworld-ubuntu"
    url: str | None = None
    token_file: str | None = None
    task: DrumsTaskConfig = DrumsTaskConfig()


class DrumsTaskset(vf.Taskset[DrumsTask, DrumsTasksetConfig]):
    def load(self) -> Iterator[DrumsTask]:
        cfg = self.config
        for tid in cfg.task_ids:
            if cfg.tasks_dir:
                hits = [f for f in os.listdir(cfg.tasks_dir) if f.startswith(tid) and f.endswith(".json")]
                if len(hits) != 1:
                    raise FileNotFoundError(f"task {tid!r}: {len(hits)} matches in {cfg.tasks_dir}")
                with open(os.path.join(cfg.tasks_dir, hits[0])) as f:
                    osw = json.load(f)
            else:
                osw = sdk.client(cfg.url, cfg.token_file).task(tid)
            yield DrumsTask(DrumsTaskData(id=osw["id"], name=osw["id"][:8], prompt=None, osworld=osw, drums_image=cfg.image),
                            cfg.task)


class DrumsEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, runtime={"type": "subprocess"})
    fork_n: int = Field(0, ge=0)
    """0: one env per episode. N > 0: one parent env, `prefix`, then N forked children, one agent each (a GRPO group)."""
    prefix: list[str] = Field(default_factory=list)
    """pyautogui code run on the parent before the fork (the reached state the group starts from)."""
    settle_s: float = 3.0
    """Seconds between the last prefix action and the fork (lets exiting processes finish before the capture)."""
    record_path: str | None = None
    """Append one JSON line per scored trace (default $DRUMS_RECORD, else no record)."""
    max_concurrent_agents: int | None = Field(8, ge=1)


def _first_message(instruction: str, png: bytes, obs) -> list:
    return [vf.UserMessage(content=[
        vf.TextContentPart(text=f"Task: {instruction}"),
        vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=A.data_url(A.observation(png, obs)))),
    ])]


class DrumsEnv(vf.Env[DrumsEnvConfig]):
    def _client(self):
        tc = self.taskset.config
        return sdk.client(tc.url, tc.token_file)

    def _record(self, row: dict) -> None:
        path = self.config.record_path or os.environ.get("DRUMS_RECORD")
        if not path:
            return
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    async def _play(self, task: DrumsTask, agents, env, parent: str | None, episode_id: str, extra: dict) -> None:
        tools = task.config.tools
        obs = (tools.obs_width, tools.obs_height)
        png = await asyncio.to_thread(env.screenshot)
        steps0 = await asyncio.to_thread(step_count, env)
        child = task.with_data(drums_steps=steps0,
            prompt=_first_message(task.data.osworld.get("instruction", ""), png, obs),
            system_prompt=SYSTEM_PROMPT.format(w=obs[0], h=obs[1]),
            drums_env_id=env.id, drums_episode=env.record.get("episode"), drums_parent=parent)
        t0 = time.time()
        trace = await agents.agent.run(child)
        d = trace.info.get("drums") or {}
        self._record({"t": time.time(), "episode_id": episode_id, "task": task.data.osworld.get("id"), "env": env.id,
                      "parent": parent, "drums_episode": env.record.get("episode"), "valid": d.get("valid"),
                      "verdict": d.get("verdict"), "failed_gates": d.get("failed_gates"), "score": d.get("score"),
                      "checker_error": d.get("checker_error"), "reward": trace.reward if (trace.ok and not trace.is_dropped) else None,
                      "rewards": {k: (v.value if v is not None else None) for k, v in trace.rewards.items()},
                      "dropped": trace.dropped, "ok": trace.ok, "errors": [e.type + ": " + e.message[:300] for e in trace.errors],
                      "turns": trace.num_turns, "actions": d.get("actions"), "stop": trace.stop_condition,
                      "agent_s": round(time.time() - t0, 3), **extra})

    async def run(self, task: DrumsTask, agents) -> None:
        c = self._client()
        episode_id = uuid.uuid4().hex[:12]
        osw = task.data.osworld
        made = []
        try:
            t0 = time.monotonic()
            parent = await asyncio.to_thread(c.create, task.data.drums_image, osw)
            made.append(parent)
            await asyncio.to_thread(parent.wait_valid)
            ready_s = time.monotonic() - t0
            if not self.config.fork_n:
                await self._play(task, agents, parent, None, episode_id, {"mode": "single", "ready_s": round(ready_s, 3)})
                return
            for code in self.config.prefix:
                await asyncio.to_thread(parent.step, {"type": "pyautogui", "code": code})
            await asyncio.sleep(self.config.settle_s)
            import drums
            t1 = time.monotonic()
            failed = []
            try:
                kids = await asyncio.to_thread(parent.fork, self.config.fork_n)
            except drums.ForkError as e:        # usable children still play; the failed ones are recorded, never scored
                kids, failed = e.ok, e.failed
            fork_s = time.monotonic() - t1
            made.extend(kids)
            for i, f in enumerate(failed):
                self._record({"t": time.time(), "episode_id": episode_id, "task": osw.get("id"), "parent": parent.id,
                              "env": getattr(f, "env_id", None), "fork_child_failed": True, "error": str(f)[:500],
                              "verdict": (getattr(f, "record", {}) or {}).get("validity", {}).get("verdict"),
                              "reward": None, "mode": "fork"})
            if not kids:
                raise RuntimeError(f"fork({self.config.fork_n}) gave no usable child: {[str(f)[:200] for f in failed]}")
            extra = {"mode": "fork", "fork_n": self.config.fork_n, "fork_ok": len(kids), "fork_failed": len(failed),
                     "fork_s": round(fork_s, 3), "ready_s": round(ready_s, 3)}
            await asyncio.gather(*[self._play(task, agents, k, parent.id, episode_id, extra) for k in kids])
        finally:
            states = {str(k.from_state) for k in made[1:]}
            for e in made:
                try:
                    await asyncio.to_thread(e.close)
                except Exception:  # noqa: BLE001
                    pass
            for st in states:
                try:
                    await asyncio.to_thread(c.delete_state, st)
                except Exception:  # noqa: BLE001
                    pass
