"""What every trainer adapter shares: one Drums desktop per episode, hybrid tool calls parsed from the policy's text,
and a reward that carries its gate verdict and audit.

A policy's turn is text. It may contain one tool call, in either form:
    <tool_call>{"name": "shell", "arguments": {"command": "ls ~"}}</tool_call>      (Qwen / Hermes style)
    {"name": "gui", "arguments": {"code": "pyautogui.click(10, 10)"}}               (a bare JSON object)
A turn with no tool call, or the word DONE outside a tool call, ends the episode. Tool names are drums.hybrid.TOOLS.

The reward an adapter hands the trainer:
    score     the checker's 0/1 when the episode is valid and its audit is clean (trusted)
    None      when the reward cannot be trusted (invalid episode, checker error, an audit flag); each adapter maps None to
              its trainer's own way of leaving a sample out (or to 0 with a mask flag where the trainer has no such way)
"""
from __future__ import annotations

import json
import os
import re

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_call(text: str):
    """-> (name, arguments) or None."""
    m = TOOL_RE.search(text or "")
    blob = m.group(1) if m else None
    if blob is None:
        s = (text or "").strip()
        if s.startswith("{") and s.endswith("}"):
            blob = s
    if blob is None:
        return None
    try:
        d = json.loads(blob)
    except ValueError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("name"), str):
        return None
    args = d.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return d["name"], args


def connect():
    import drums
    url = os.environ.get("DRUMS_URL") or "http://127.0.0.1:7070"
    tf = os.environ.get("DRUMS_TOKEN_FILE") or os.path.expanduser("~/.drums-token")
    tok = open(tf).read().strip() if os.path.exists(tf) else os.environ.get("DRUMS_TOKEN")
    return drums.connect(url, tok)


class Episode:
    """One task on one fresh desktop. step(text) -> (observation text, done); finish() -> reward dict."""

    def __init__(self, client, task: dict, max_turns: int = 15):
        from drums.hybrid import HybridTools
        self.client, self.task, self.max_turns = client, task, max_turns
        self.env = client.create(task=task, idle_timeout=900)
        self.env.wait_valid(timeout=120)
        self.tools = HybridTools(self.env)
        self.turns = 0
        self.calls = []

    def prompt(self) -> str:
        names = ", ".join(t["name"] for t in __import__("drums.hybrid", fromlist=["TOOLS"]).TOOLS)
        return (f"{self.task['instruction']}\n\nYou control an Ubuntu desktop. Call one tool per turn as "
                f"<tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call>. Tools: {names}. Say DONE when finished.")

    def step(self, text: str):
        self.turns += 1
        call = parse_tool_call(text)
        if call is None:
            return "", True
        name, args = call
        try:
            out = self.tools.call(name, args)
            obs = f"[screenshot: {len(out)} bytes PNG]" if isinstance(out, (bytes, bytearray)) else json.dumps(out)[:4000]
        except Exception as e:  # noqa: BLE001 - a bad call is the policy's mistake: it sees the error
            obs = f"tool error: {type(e).__name__}: {e}"[:500]
        self.calls.append(name)
        done = self.turns >= self.max_turns or ("DONE" in TOOL_RE.sub("", text))
        return obs, done

    def finish(self) -> dict:
        r = self.env.verify(strict=False)
        trusted = bool(r.valid and r.score is not None and r.checker_error is None and not r.audit_flags)
        out = {"reward": float(r.score) if trusted else None, "score": r.score, "valid": r.valid,
               "checker_error": r.checker_error, "audit_flags": r.audit_flags, "trusted": trusted, "audit": r.audit,
               "env": r.env_id, "episode": r.episode, "turns": self.turns, "tools_used": self.calls}
        return out

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass


def oracle_turns(sol: dict) -> list:
    """The scripted policy: a task's oracle actions as hybrid tool calls (text turns), then DONE."""
    turns = []
    for a in sol["oracle"]:
        if a["type"] == "pyautogui":
            call = {"name": "gui", "arguments": {"code": a["code"]}}
        elif a["type"] == "sleep":
            call = {"name": "shell", "arguments": {"command": f"sleep {a['seconds']}"}}
        else:
            import shlex
            cmd = a["command"] if isinstance(a["command"], str) else " ".join(shlex.quote(x) for x in a["command"])
            call = {"name": "shell", "arguments": {"command": cmd, "timeout_s": 120}}
        turns.append(f"<tool_call>{json.dumps(call)}</tool_call>")
    return turns + ["DONE"]


def probe_turns() -> list:
    """A policy that looks and gives up: a valid failure (reward 0)."""
    return ['<tool_call>{"name": "screenshot", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "list_dir", "arguments": {"path": "~"}}</tool_call>', "DONE"]
