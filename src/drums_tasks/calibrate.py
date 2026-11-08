"""Difficulty calibration: the L1 policy (ScaleCUA 9B, the checkpoint L1 trained from) attempts factory tasks K times;
each task gets a pass rate, and a curriculum can target tasks near 0.5 (an all-pass or all-fail GRPO group has no gradient).

    python -m drums_tasks.calibrate --tasks DIR --out OUT --model-url http://HOST:8000/v1 [--per-family 3] [--k 4]
                                    [--workers 16] [--max-turns 15] [--sample-seed 20261108]

The agent is the ScaleCUA tool format from drums_verifiers.scalecua (system prompt, the single `computer` tool, 0..1000
relative coordinates, action -> pyautogui), GUI only (no shell for the policy), 5 newest screenshots kept, temperature
0.8, 1,536 output tokens, a 2 s pause before each screenshot: the L1 settings. Reward = the task's checker on the final
state, counted only when the episode is valid. Writes OUT/episodes.jsonl and OUT/calibration.json (per-task pass rate,
the 0.5-band candidates, and per-family means).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import random
import sys
import threading
import time
import urllib.request

import drums

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_scalecua():
    """drums_verifiers/scalecua.py by path (stdlib only): the package's __init__ needs verifiers, this does not."""
    import importlib.util
    for p in (os.path.join(HERE, "..", "drums_verifiers", "scalecua.py"), os.path.join(HERE, "scalecua.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("t6_scalecua", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise ImportError("drums_verifiers/scalecua.py not found next to drums_tasks")


S = _load_scalecua()

LOCK = threading.Lock()
OBS = (1280, 720)


def obs_png(png: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB").resize(OBS, Image.Resampling.LANCZOS)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def img_part(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(obs_png(png)).decode()}}


def keep_images(msgs, k=5):
    idx = [i for i, m in enumerate(msgs) if m["role"] == "tool" and isinstance(m.get("content"), list)
           and any(p.get("type") == "image_url" for p in m["content"])]
    out = [dict(m) for m in msgs]
    for i in idx[:-k] if k else []:
        out[i]["content"] = [p for p in out[i]["content"] if p.get("type") != "image_url"]
    return out


def chat(url, msgs, tools, max_tokens, temperature, timeout=300):
    body = {"model": "scalecua", "messages": msgs, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens,
            "temperature": temperature}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def episode(c, task, a):
    rec = {"task": task["id"], "t0": time.time()}
    env = None
    try:
        env = c.create(task=task, idle_timeout=1800)
        env.wait_valid(timeout=180)
        tools = S.make_computer_tools(True, OBS)
        msgs = [{"role": "system", "content": S.build_system_prompt()},
                {"role": "user", "content": [img_part(env.screenshot()), {"type": "text", "text": task["instruction"]}]}]
        turns, acts, stop = 0, 0, "max_turns"
        for turns in range(1, a.max_turns + 1):
            r = chat(a.model_url, keep_images(msgs), tools, a.max_tokens, a.temperature)
            m = r["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if "[INFEASIBLE]" in (m.get("content") or ""):
                stop = "infeasible"
                break
            if not calls:
                stop = "no_tool_call"
                break
            done = False
            for tc in calls:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    code = S.args_to_pyautogui(args, 1920, 1080, True)
                except (ValueError, TypeError, KeyError) as e:
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": f"Invalid action: {e}"})
                    continue
                if code in (S.DONE, S.FAIL, "CALL_USER"):
                    done = True
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": "Done"})
                    break
                env.step({"type": "pyautogui", "code": code}, check=False)
                acts += 1
                time.sleep(a.pause_s)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": [{"type": "text", "text": "Success"}, img_part(env.screenshot())]})
            if done:
                stop = "done"
                break
        rew = env.verify(strict=False)
        rec.update(score=rew.score, valid=rew.valid, trusted=rew.trusted, audit_flags=rew.audit_flags, turns=turns,
                   actions=acts, stop=stop, checker_error=rew.checker_error)
    except Exception as e:  # noqa: BLE001 - an infrastructure or model-server failure is not a policy failure
        rec.update(score=None, valid=False, error=f"{type(e).__name__}: {e}"[:400])
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
    rec["wall_s"] = round(time.time() - rec["t0"], 1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-url", required=True)
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--only-accepted", default="", help="records.jsonl of a validation run: calibrate accepted tasks only")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--pause-s", type=float, default=2.0)
    ap.add_argument("--sample-seed", type=int, default=20261108)
    ap.add_argument("--deadline-min", type=float, default=0)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.drums-token"))
    a = ap.parse_args(argv)
    tasks = [json.loads(x) for x in open(os.path.join(a.tasks, "tasks.jsonl"))]
    fam = {json.loads(x)["id"]: json.loads(x)["family"] for x in open(os.path.join(a.tasks, "solutions.jsonl"))}
    ok = None
    if a.only_accepted:
        ok = {r["id"] for r in (json.loads(x) for x in open(a.only_accepted)) if r["accepted"]}
    by = {}
    for t in tasks:
        if ok is None or t["id"] in ok:
            by.setdefault(fam[t["id"]], []).append(t)
    rnd = random.Random(a.sample_seed)
    pick = [t for f in sorted(by) for t in sorted(rnd.sample(by[f], min(a.per_family, len(by[f]))), key=lambda t: t["id"])]
    jobs = [(t, k) for k in range(a.k) for t in pick]          # round-robin over k: a deadline cuts evenly
    os.makedirs(a.out, exist_ok=True)
    ep = os.path.join(a.out, "episodes.jsonl")
    c = drums.connect(token=open(a.token_file).read().strip())
    deadline = time.time() + a.deadline_min * 60 if a.deadline_min else None
    print(f"{len(pick)} tasks x k={a.k} = {len(jobs)} episodes, {a.workers} workers", flush=True)

    def one(job):
        t, k = job
        if deadline and time.time() > deadline:
            return
        r = episode(c, t, a)
        r.update(k=k, family=fam[t["id"]])
        with LOCK:
            with open(ep, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps({x: r.get(x) for x in ("task", "k", "score", "valid", "turns", "stop", "wall_s", "error")}), flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))
    eps = [json.loads(x) for x in open(ep)]
    per = {}
    for e in eps:
        if e.get("valid") and e.get("score") is not None:
            per.setdefault(e["task"], []).append(e["score"])
    rate = {t: sum(v) / len(v) for t, v in per.items()}
    fams = {}
    for t, r in rate.items():
        fams.setdefault(fam[t], []).append(r)
    out = {"model": "ScaleCUA-qwen3.5-osworld-sft ff2508f6 (the L1 step-0 policy)", "k": a.k, "tasks": len(rate),
           "valid_episodes": sum(len(v) for v in per.values()), "episodes": len(eps),
           "pass_rate": rate, "n_valid": {t: len(v) for t, v in per.items()},
           "all_fail": sorted(t for t, r in rate.items() if r == 0.0), "all_pass": sorted(t for t, r in rate.items() if r == 1.0),
           "band_0.25_0.75": sorted(t for t, r in rate.items() if 0.25 <= r <= 0.75),
           "family_mean": {f: round(sum(v) / len(v), 3) for f, v in sorted(fams.items())}}
    json.dump(out, open(os.path.join(a.out, "calibration.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("tasks", "valid_episodes", "episodes")}), "all_fail", len(out["all_fail"]),
          "all_pass", len(out["all_pass"]), "band", len(out["band_0.25_0.75"]))


if __name__ == "__main__":
    main()
