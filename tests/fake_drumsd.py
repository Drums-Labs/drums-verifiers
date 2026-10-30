"""A fake drumsd (the HTTP routes the SDK calls) for offline tests: envs are dicts, the desktop is a list of actions, the
"checker" scores 1.0 when the expected text was typed. Controls:
  an action containing INVALIDATE        -> the env's verdict becomes invalid (failed gate "alive_after_resume")
  a fork after a prefix with KILL_CHILDREN -> every child's verdict is invalid (the renderer-exit case of the SDK v0 report)
  an action containing CRASH             -> HTTP 502 (guest error)
  an action containing QUARANTINE        -> the env is quarantined (invalid) and the step gets HTTP 409, as drumsd does
  creating an env for task fa11ed00-...  -> the slot comes back down (restore failed: the crashed-restore case)
  a fork after a prefix with OVERLOAD_TWICE -> the first two fork requests of that env get HTTP 429 overloaded (memory
                                         admission, nothing started), the third succeeds; OVERLOAD_ALWAYS -> always 429
    python fake_drumsd.py PORT TOKEN"""
from __future__ import annotations

import base64
import io
import itertools
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _png(w=1920, h=1080):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), (40, 20, 60)).save(b, format="PNG")
    return b.getvalue()


FAIL_TASK = "fa11ed00-0000-4000-8000-000000000000"
TASKS = {
    FAIL_TASK: {"id": FAIL_TASK, "instruction": "never starts", "config": [], "evaluator": {"func": "fake", "expected": "x"}},
    "e0df059f-28a6-4169-924f-b9623e7184cc": {"id": "e0df059f-28a6-4169-924f-b9623e7184cc",
                                             "instruction": "Rename the folder todo_list_Jan_1 to todo_list_Jan_2.",
                                             "config": [], "evaluator": {"func": "fake", "expected": "todo_list_Jan_2"}},
}


class State:
    def __init__(self):
        self.envs, self.ck, self.lock, self.n = {}, {}, threading.Lock(), itertools.count(101)
        self.png = _png()
        self.calls = []

    def new(self, checkpoint, task, actions=(), invalid=False):
        eid = f"s{next(self.n)}"
        self.envs[eid] = {"id": eid, "state": "ready", "episode": 1, "checkpoint": checkpoint, "task": task,
                          "actions": list(actions), "base": list(actions), "invalid": invalid, "steps": 0}
        return self.envs[eid]

    def rec(self, e):
        v = "invalid" if e["invalid"] else "valid"
        return {"id": e["id"], "state": e["state"], "episode": e["episode"], "checkpoint": e["checkpoint"], "reason": e.get("reason"),
                "task": ({"id": e["task"]["id"], "instruction": e["task"]["instruction"]} if e["task"] else None),
                "validity": {"verdict": v, "gates": ["alive_after_resume"] if e["invalid"] else []}}


S = State()


def make_handler(token):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, raw=None, ctype="application/json"):
            b = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _auth(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "bad token"})
                return False
            return True

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if not self._auth():
                return
            p = self.path.split("?")[0].rstrip("/").split("/")[1:]
            S.calls.append(("GET", self.path))
            if p == ["v1", "health"]:
                return self._send(200, {"ok": True, "version": "fake"})
            if p[:2] == ["v1", "tasks"] and len(p) == 3:
                t = next((v for k, v in TASKS.items() if k.startswith(p[2])), None)
                return self._send(200, t) if t else self._send(404, {"error": "no task"})
            if p[:2] == ["v1", "slots"] and len(p) >= 3:
                e = S.envs.get(p[2])
                if e is None:
                    return self._send(404, {"error": "no env"})
                if len(p) == 3:
                    return self._send(200, S.rec(e))
                if p[3] == "screenshot":
                    return self._send(200, None, S.png, "image/png")
                if p[3] == "gates":
                    return self._send(200, {"episode": e["episode"], "validity": S.rec(e)["validity"]})
                if p[3] == "metrics":
                    return self._send(200, {"episode": e["episode"], "usage": {"ops": {"step": {"n": e["steps"]}}}})
            self._send(404, {"error": "route"})

        def do_DELETE(self):
            if not self._auth():
                return
            p = self.path.rstrip("/").split("/")[1:]
            S.calls.append(("DELETE", self.path))
            if p[:2] == ["v1", "slots"]:
                e = S.envs[p[2]]
                e["state"] = "destroyed"
                return self._send(200, S.rec(e))
            if p[:2] == ["v1", "checkpoints"]:
                S.ck.pop(p[2], None)
                return self._send(200, {"deleted": p[2]})
            self._send(404, {"error": "route"})

        def do_POST(self):
            if not self._auth():
                return
            p = self.path.rstrip("/").split("/")[1:]
            body = self._body()
            S.calls.append(("POST", self.path, json.dumps(body)[:200]))
            with S.lock:
                if p == ["v1", "slots"]:
                    if (body.get("task") or {}).get("id") == FAIL_TASK:     # as drumsd: the slot exists, down, with the reason
                        e = S.new(body.get("checkpoint"), body.get("task"))
                        e["state"], e["reason"] = "down", "criu restore failed (exit None)"
                        return self._send(200, S.rec(e))
                    ck = body.get("checkpoint")
                    if ck in S.ck:
                        base, inv, task = S.ck[ck]
                        e = S.new(ck, task, base, inv)
                    else:
                        e = S.new(ck, body.get("task"))
                    return self._send(200, S.rec(e))
                e = S.envs.get(p[2])
                if e is None or e["state"] == "destroyed":
                    return self._send(404, {"error": "no env"})
                op = p[3]
                if op == "reset":
                    e["actions"], e["episode"] = list(e["base"]), e["episode"] + 1
                    return self._send(200, S.rec(e))
                if op == "step":
                    a = body["action"]
                    code = a.get("code") or json.dumps(a.get("command"))
                    if "CRASH" in code:
                        return self._send(502, {"error": "guest down"})
                    if e["state"] == "quarantined":
                        return self._send(409, {"error": "slot is quarantined: step is not allowed (allowed from: ready)"})
                    if "INVALIDATE" in code:
                        e["invalid"] = True
                    if "QUARANTINE" in code:
                        e["invalid"], e["state"], e["reason"] = True, "quarantined", "alive_after_resume"
                        return self._send(409, {"error": "slot is quarantined: step is not allowed (allowed from: ready)"})
                    e["actions"].append(code)
                    e["steps"] += 1
                    r = {"result": {"status": "success", "output": "", "error": "", "returncode": 0}, "seconds": 0.01,
                         "episode": e["episode"], "validity": S.rec(e)["validity"]}
                    if body.get("screenshot"):
                        r["screenshot_png_b64"] = base64.b64encode(S.png).decode()
                    return self._send(200, r)
                if op == "verify":
                    t = e["task"] or {}
                    want = (t.get("evaluator") or {}).get("expected")
                    import re
                    typed = ["".join(re.findall(r"press\('(.)'\)", a)) for a in e["actions"]]   # ScaleCUA types key by key
                    score = 1.0 if want and any(want in a for a in e["actions"] + typed) else 0.0
                    rec = S.rec(e)
                    return self._send(200, {"slot": e["id"], "episode": e["episode"], "task_id": t.get("id"), "score": score,
                                            "valid": not e["invalid"], "validity": rec["validity"], "checker_error": None,
                                            "gates": {}, "seconds": 0.01})
                if op == "branch":
                    e["fork_tries"] = e.get("fork_tries", 0) + 1
                    if any("OVERLOAD_ALWAYS" in a for a in e["actions"]) or (any("OVERLOAD_TWICE" in a for a in e["actions"]) and e["fork_tries"] <= 2):
                        return self._send(429, {"error": "overloaded: fork refused: 1000 MiB available", "reason": "memory",
                                                "admission": {"available_mib": 1000, "needed_mib": 6144, "floor_mib": 4096}})
                    ck = f"ck-{e['id']}-{len(S.ck)}"
                    kill = any("KILL_CHILDREN" in a for a in e["actions"])
                    S.ck[ck] = (list(e["actions"]), e["invalid"] or kill, e["task"])
                    kids = [S.rec(S.new(ck, e["task"], e["actions"], e["invalid"] or kill)) for _ in range(int(body["n"]))]
                    return self._send(200, {"children": kids, "checkpoint": {"id": ck}})
                if op == "capture":
                    ck = f"ck-{e['id']}-{len(S.ck)}"
                    S.ck[ck] = (list(e["actions"]), e["invalid"], e["task"])
                    return self._send(200, {"id": ck})
            self._send(404, {"error": "route"})
    return H


def serve(port, token):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), make_handler(sys.argv[2]))
    srv.serve_forever()
