"""A scripted stand-in for the model: an OpenAI-compatible /v1/chat/completions server that answers with computer_* tool
calls. It lets the whole rollout path (orchestrator -> env server -> harness -> tools -> Drums) run on CPU, with no GPU and
no model. Not a policy: it picks a plan per conversation and plays it.

Plans (chosen round-robin per new conversation, written into the first reply as "plan=<name>" so later turns follow it):
  solve-e0df   the e0df059f rename, by GUI: click the folder icon, F2, ctrl+a, type the new name, Enter
  probe        screenshot, a seeded mouse move and click, Escape
  crash        types the marker CRASH (the test fake drumsd answers it with a guest error: an infrastructure failure)
  quarantine   types the marker QUARANTINE (the test fake drumsd quarantines the env: a gate failure mid-episode)
then a final reply without a tool call (the harness ends the rollout).

    python mock_policy.py --port 8011 [--plans solve-e0df,probe] [--icon 1235 590] [--log calls.jsonl]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLANS = {}


def plan_steps(name, icon, rnd):
    if name == "solve-e0df":
        x, y = icon
        return [("computer_click", {"x": x, "y": y}), ("computer_key", {"keys": "f2"}), ("computer_key", {"keys": "ctrl+a"}),
                ("computer_type_text", {"text": "todo_list_Jan_2"}), ("computer_key", {"keys": "enter"}),
                ("computer_screenshot", {})]
    if name in ("crash", "quarantine"):
        return [("computer_type_text", {"text": name.upper()}), ("computer_screenshot", {})]
    x, y = rnd.randint(300, 900), rnd.randint(200, 500)
    return [("computer_screenshot", {}), ("computer_move", {"x": x, "y": y}), ("computer_click", {"x": x, "y": y}),
            ("computer_key", {"keys": "escape"})]


class Policy:
    def __init__(self, plans, icon, log):
        self.plans, self.icon, self.log = plans, icon, log
        self.counter = itertools.count()
        self.lock = threading.Lock()

    def reply(self, body):
        msgs = body.get("messages") or []
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        first_user = next((m for m in msgs if m.get("role") == "user"), {})
        text = json.dumps(first_user.get("content"))[:4000]
        if assistants:
            m = re.search(r"plan=([\w-]+) seed=(\d+)", str(assistants[0].get("content") or ""))
            plan, seed = (m.group(1), int(m.group(2))) if m else ("probe", 0)
        else:
            with self.lock:
                n = next(self.counter)
            plan = self.plans[n % len(self.plans)]
            if plan == "solve-e0df" and "todo_list" not in text and "folder" not in text.lower():
                plan = "probe"             # only the rename task has a scripted solution
            seed = n
        steps = plan_steps(plan, self.icon, random.Random(seed))
        i = len(assistants)
        note = f"plan={plan} seed={seed} step={i}"
        if i < len(steps):
            name, args = steps[i]
            msg = {"role": "assistant", "content": note,
                   "tool_calls": [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                   "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": note + " done."}
            finish = "stop"
        if self.log:
            with self.lock, open(self.log, "a") as f:
                f.write(json.dumps({"t": time.time(), "plan": plan, "seed": seed, "turn": i, "n_messages": len(msgs),
                                    "tools": len(body.get("tools") or []), "reply": msg}) + "\n")
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--plans", default="solve-e0df,probe")
    ap.add_argument("--icon", type=int, nargs=2, default=[1235, 590], help="the e0df folder icon in 1280x720 screenshot pixels")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    pol = Policy(a.plans.split(","), tuple(a.icon), a.log)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _stream(self, r, body):
            """The same reply as server-sent chat.completion.chunk events (what streaming harnesses ask for)."""
            msg, ch = r["choices"][0]["message"], r["choices"][0]
            base = {k: r[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
            delta = {"role": "assistant", "content": msg.get("content") or ""}
            events = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}]
            for i, tcall in enumerate(msg.get("tool_calls") or []):
                events.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **tcall}]}, "finish_reason": None}]})
            events.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": ch["finish_reason"]}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append({**base, "choices": [], "usage": r["usage"]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev in events:
                self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path.rstrip("/").endswith("/chat/completions"):
                r = pol.reply(body)
                if body.get("stream"):
                    return self._stream(r, body)
                return self._send(200, r)
            self._send(404, {"error": "not found"})

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
