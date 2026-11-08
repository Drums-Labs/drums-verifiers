"""End-to-end proof of the verl and SkyRL adapters on a live drumsd host, with a scripted policy.

    python -I t6/adapters_e2e.py --tasks TS_DIR --out OUT [--per-family 1] [--families a,b] [--adapters skyrl,verl]

For each picked task, each adapter runs two rollouts through the trainer library's own interface:
  oracle   the task's scripted oracle, sent as hybrid tool calls (expected reward 1.0)
  probe    a screenshot, a directory listing, then DONE (expected reward 0.0)
SkyRL: skyrl_gym.make("drums", extras={"drums_task": ...}) then init()/step() exactly as SkyRL's generator drives an env.
verl:  verl's own ToolAgentLoop machinery (DrumsToolAgentLoop subclass, hermes tool parser, Qwen2.5 tokenizer, token-in
       token-out merging) with a scripted stand-in for the LLM server that returns the policy's turns as token ids.
Writes OUT/adapters.jsonl (one line per rollout) and OUT/adapters-summary.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.expanduser("~/t6"))

from drums_adapters.common import oracle_turns, probe_turns  # noqa: E402

DEFAULT_FAMILIES = "files_rename,files_move_ext,files_count,chrome_download,multi_download_extract,files_mkdir_tree"


def run_skyrl(task, turns):
    import skyrl_gym
    from skyrl_gym.envs import register
    try:
        register(id="drums", entry_point="drums_adapters.skyrl_env:DrumsEnv")
    except Exception:  # noqa: BLE001 - already registered
        pass
    env = skyrl_gym.make("drums", env_config={"max_turns": 40}, extras={"drums_task": task})
    t0 = time.time()
    obs, info = env.init([])
    out = None
    for t in turns:
        out = env.step(t)
        if out["done"]:
            break
    env.close()
    return {"reward": out["reward"], "done": out["done"], "metadata": out["metadata"], "init_info": info, "wall_s": round(time.time() - t0, 2)}


class ScriptedServer:
    """Stands in for verl's LLM server manager: returns the next scripted turn for the trajectory as token ids."""

    def __init__(self, tok, scripts):
        self.tok, self.scripts, self.pos = tok, scripts, {}

    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        from verl.workers.rollout.replica import TokenOutput
        if request_id not in self.pos:
            text = self.tok.decode(prompt_ids)
            key = next(k for k in self.scripts if k in text)
            self.pos[request_id] = [key, 0]
        key, i = self.pos[request_id]
        turns = self.scripts[key]
        self.pos[request_id][1] += 1
        t = turns[min(i, len(turns) - 1)]
        if t.startswith("<tool_call>"):
            body = json.loads(t[len("<tool_call>"):-len("</tool_call>")])
            t = "<tool_call>\n" + json.dumps({"name": "drums_desktop", "arguments": body}) + "\n</tool_call>"
        ids = self.tok.encode(t, add_special_tokens=False) + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        return TokenOutput(token_ids=ids, log_probs=[0.0] * len(ids), stop_reason="completed", num_preempted=0)


EV = None


def verl_loop(scripts):
    global EV
    EV = asyncio.new_event_loop()          # verl's AgentLoopBase binds the loop current at construction
    asyncio.set_event_loop(EV)
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    import verl
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from drums_adapters.verl_tool import DrumsDesktopTool, DrumsToolAgentLoop
    cfg = OmegaConf.load(os.path.join(os.path.dirname(verl.__file__), "trainer/config/_generated_ppo_trainer.yaml"))
    r = cfg.actor_rollout_ref.rollout
    r.multi_turn.enable = True
    r.multi_turn.format = "hermes"
    r.multi_turn.max_assistant_turns = 40
    r.multi_turn.max_user_turns = 40
    r.multi_turn.max_tool_response_length = 4000
    r.prompt_length = 8192
    r.response_length = 200000
    tok = AutoTokenizer.from_pretrained(os.environ.get("T6_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct"))
    tool = DrumsDesktopTool({"type": "native", "max_turns": 40, "images": False})   # the stand-in tokenizer is text-only
    loop = DrumsToolAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=ScriptedServer(tok, scripts), tokenizer=tok,
                              processor=None, dataset_cls=RLHFDataset, data_config=DictConfigWrap(cfg.data), tools=ToolListWrap([tool]))
    return loop


def run_verl(loop, task, key):
    t0 = time.time()
    out = EV.run_until_complete(loop.run({"temperature": 1.0}, raw_prompt=[{"role": "user", "content": f"[{key}] {task['instruction']}"}],
                               extra_info={"drums_task": task, "max_turns": 40}, tools_kwargs={}))
    ef = {k: v for k, v in out.extra_fields.items() if k.startswith("drums_")}
    return {"reward": out.reward_score, "num_turns": out.num_turns, "response_tokens": len(out.response_ids),
            "tool_tokens_masked": out.response_mask.count(0), "extra": ef, "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=1)
    ap.add_argument("--families", default=DEFAULT_FAMILIES)
    ap.add_argument("--adapters", default="skyrl,verl")
    a = ap.parse_args()
    tasks = {}
    for line in open(os.path.join(a.tasks, "tasks.jsonl")):
        t = json.loads(line)
        tasks[t["id"]] = t
    sols = [json.loads(x) for x in open(os.path.join(a.tasks, "solutions.jsonl"))]
    fams = a.families.split(",")
    pick = []
    for f in fams:
        pick += [s for s in sols if s["family"] == f][:a.per_family]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    scripts = {}
    for s in pick:
        scripts[f"oracle:{s['id']}"] = oracle_turns(s)
        scripts[f"probe:{s['id']}"] = probe_turns()
    adapters = a.adapters.split(",")
    loop = verl_loop(scripts) if "verl" in adapters else None
    for s in pick:
        task = tasks[s["id"]]
        for policy in ("oracle", "probe"):
            turns = scripts[f"{policy}:{s['id']}"]
            for ad in adapters:
                t0 = time.time()
                try:
                    res = run_skyrl(task, turns) if ad == "skyrl" else run_verl(loop, task, f"{policy}:{s['id']}")
                    err = None
                except Exception as e:  # noqa: BLE001
                    import traceback
                    res, err = {}, traceback.format_exc()[-1500:]
                row = {"adapter": ad, "task": s["id"], "family": s["family"], "policy": policy,
                       "expected": 1.0 if policy == "oracle" else 0.0, "error": err, **res}
                row["pass"] = err is None and res.get("reward") == row["expected"]
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("adapter", "task", "policy", "reward", "pass", "wall_s")}), flush=True)
                with open(os.path.join(a.out, "adapters.jsonl"), "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    summ = {}
    for ad in adapters:
        rs = [r for r in rows if r["adapter"] == ad]
        summ[ad] = {"rollouts": len(rs), "pass": sum(r["pass"] for r in rs), "errors": sum(1 for r in rs if r["error"])}
    json.dump(summ, open(os.path.join(a.out, "adapters-summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
