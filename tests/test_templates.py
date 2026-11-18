"""The bring-your-own-GPUs templates (templates/): the shared runner renders valid prime-rl TOMLs from the one config,
moves partial checkpoints aside before a resume, keeps the spend ledger and stops at the budget, writes refs/main for a
pinned download, and reports a pool's expected zero-advantage share. Offline: no GPU, no Modal, no Drums host."""
import importlib.util
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
spec = importlib.util.spec_from_file_location("drums_rl", TEMPLATES / "common" / "drums_rl.py")
R = importlib.util.module_from_spec(spec)
sys.modules["drums_rl"] = R
spec.loader.exec_module(R)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUMS_RL_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("DRUMS_RL_WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.delenv("DRUMS_RL_PLATFORM", raising=False)
    split = tmp_path / "split"
    split.mkdir()
    rows = [{"id": f"t6-demo-{i:04d}", "instruction": f"do {i}", "config": [], "evaluator": {}} for i in range(5)]
    (split / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, split, [r["id"] for r in rows]


def cfg():
    return R.load_config(TEMPLATES / "rl-config.toml")


def test_config_is_d1_and_checks_the_split(tmp_path):
    c = cfg()
    assert c["model"]["repo"] == "Qwen/Qwen3.5-27B" and len(c["model"]["revision"]) == 40
    assert c["gpus"]["modal"] == ["H200:8", "B200:8"] and c["gpus"]["inference"] + c["gpus"]["trainer"] == 8
    bad = tmp_path / "bad.toml"
    bad.write_text((TEMPLATES / "rl-config.toml").read_text().replace("trainer = 6", "trainer = 5"))
    with pytest.raises(ValueError, match="must equal count"):
        R.load_config(bad)


def test_train_toml_renders_to_the_d1_recipe_plus_the_d2_levers(env):
    tmp, split, ids = env
    ids2 = R.explode_tasks(split, tmp / "json")
    assert ids2 == ids and sorted(os.listdir(tmp / "json")) == [f"{i}.json" for i in ids]
    text = R.render_train(cfg(), ids, tmp / "json")
    assert "@" not in "".join(line.split("#")[0] for line in text.splitlines())
    t = tomllib.loads(text)
    assert t["model"]["name"] == "Qwen/Qwen3.5-27B" and t["max_steps"] == 64 and t["seq_len"] == 32768
    assert t["deployment"] == {"type": "single_node", "gpus_per_node": 8, "num_infer_gpus": 2, "num_train_gpus": 6}
    assert t["inference"]["vllm"]["tensor_parallel_size"] == 2 and t["inference"]["vllm"]["enable_prefix_caching"] is True
    assert t["trainer"]["loss"] == {"type": "ipo", "eps": 0.3} and t["trainer"]["optim"]["lr"] == 3e-6     # A-11
    assert t["orchestrator"]["batch_size"] == 128 and t["orchestrator"]["train"]["group_size"] == 8
    assert t["orchestrator"]["concurrency"] == {"initial_inflight": 128, "min_inflight": 96, "max_inflight": 128}
    assert t["orchestrator"]["renderer"]["enable_thinking"] is False
    src = t["orchestrator"]["train"]["source"][0]
    assert src["env"]["taskset"]["task_ids"] == ids and src["env"]["taskset"]["tasks_dir"] == str(tmp / "json")
    assert src["env"]["max_concurrent_agents"] == 128 and src["env"]["fork_n"] == 0
    h = src["env"]["agent"]["harness"]
    assert h == {"id": "drums-chat", "screenshots_full": 3, "screenshots_total": 6, "downscale": 0.5, "screenshots_stride": 3}
    assert src["env"]["agent"]["max_turns"] == 15 and src["env"]["taskset"]["task"]["tools"]["obs_width"] == 1280
    assert src["env"]["record_path"] == str(tmp / "runs" / "train" / "drums-record.jsonl")


def test_harness_keys_are_drums_chat_config_fields(env):
    from drums_chat import DrumsChatConfig
    if "screenshots_full" not in DrumsChatConfig.model_fields:
        pytest.skip("the screenshot-window options are not in this drums-verifiers yet")
    tmp, split, ids = env
    h = tomllib.loads(R.render_train(cfg(), ids, tmp))["orchestrator"]["train"]["source"][0]["env"]["agent"]["harness"]
    c = DrumsChatConfig(**{k: v for k, v in h.items() if k != "id"})
    assert c.window_args() == (3, 6, 0.5, 3)


def test_smoke_is_one_small_step_in_its_own_run_dir(env):
    tmp, split, ids = env
    t = tomllib.loads(R.render_train(cfg(), ids, tmp, smoke=True))
    assert t["max_steps"] == 1 and t["run"]["name"] == "train-smoke" and t["orchestrator"]["batch_size"] == 8
    assert t["orchestrator"]["concurrency"]["max_inflight"] == 8


def test_eval_runs_the_agent_exactly_as_training(env):
    tmp, split, ids = env
    inf, ev = R.render_eval(cfg(), "/runs/exports/train/step_6", ids, tmp, 6, "last-heldout", 48)
    i, e = tomllib.loads(inf), tomllib.loads(ev)
    assert i["vllm"]["model"] == "/runs/exports/train/step_6" and i["vllm"]["tensor_parallel_size"] == 2
    assert e["group_size"] == 6 and e["concurrency"] == {"min_inflight": 48, "max_inflight": 48}
    train_env = tomllib.loads(R.render_train(cfg(), ids, tmp))["orchestrator"]["train"]["source"][0]["env"]
    eval_env = e["source"][0]["env"]
    assert eval_env["agent"] == train_env["agent"] and eval_env["taskset"]["task"] == train_env["taskset"]["task"]
    assert eval_env["record_path"].endswith("eval/last-heldout/last-heldout.jsonl")


def test_render_refuses_a_missing_value():
    with pytest.raises(KeyError, match="NOPE"):
        R.render('a = "@NOPE@"', {})
    assert R.render('a = ["@L@"]\nb = "@N@"', {"L": ["x", "y"], "N": 3}) == 'a = ["x", "y"]\nb = 3'


def make_step(run: Path, n: int, complete: bool) -> Path:
    d = run / "checkpoints" / f"step_{n}"
    (d / "orchestrator").mkdir(parents=True)
    (d / "orchestrator" / "progress.pt").write_text("x")
    if complete:
        (d / "trainer").mkdir()
        (d / "trainer" / ".metadata").write_text("x")
    return d


def test_partial_step_dirs_are_moved_aside_before_a_resume(tmp_path):
    """D1: a preemption between the orchestrator's save and the trainer's left step_3 with only progress.pt; prime-rl's
    --resume took it by name and failed until it was deleted by hand."""
    run = tmp_path / "train"
    make_step(run, 1, True)
    make_step(run, 2, True)
    make_step(run, 3, False)
    assert R.latest_complete(run).name == "step_2"
    assert R.move_partial_steps(run) == ["step_3"]
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["step_1", "step_2"]
    kept = list((run / "checkpoints-partial").iterdir())
    assert len(kept) == 1 and (kept[0] / "orchestrator" / "progress.pt").exists()      # kept, never deleted
    assert R.move_partial_steps(run) == []
    make_step(run, 10, True)
    assert R.latest_complete(run).name == "step_10"                                     # numeric, not lexical


def test_refs_main_is_written_for_a_pinned_snapshot(tmp_path):
    snap = tmp_path / "hub" / "models--Qwen--Qwen3.5-27B" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    ref = R.write_refs_main(snap)
    assert ref == tmp_path / "hub" / "models--Qwen--Qwen3.5-27B" / "refs" / "main" and ref.read_text() == "a" * 40


def test_prices_regions_and_the_ledger(env, monkeypatch):
    tmp, *_ = env
    c = cfg()
    assert R.usd_per_hour(c, "NVIDIA H200", 8, 32, 256) == pytest.approx(8 * 4.54 + 32 * 0.0473 + 256 * 0.008)  # 39.88 (D1)
    c2 = {**c, "gpus": {**c["gpus"], "region": "us-east"}}
    assert R.usd_per_hour(c2, "NVIDIA B200", 8, 32, 256) == pytest.approx(1.75 * (50 + 32 * 0.0473 + 256 * 0.008))
    with pytest.raises(ValueError, match="no price"):
        R.usd_per_hour(c, "NVIDIA A10G", 1, 4, 16)
    monkeypatch.setenv("DRUMS_RL_PLATFORM", "gpu-box")
    assert R.usd_per_hour({**c, "prices": {**c["prices"], "box_usd_per_hour": 20}}, "NVIDIA H100", 8, 0, 0) == 20
    ledger = tmp / "runs" / "spend.jsonl"
    ledger.parent.mkdir(parents=True)
    now = time.time()
    rows = [{"attempt": "smoke-1", "what": "smoke", "t0": now - 3600, "t": now - 3000, "usd_per_h": 40.0},
            {"attempt": "train-1", "what": "train", "t0": now - 7200, "t": now - 3600, "usd_per_h": 40.0},
            {"attempt": "train-1", "what": "train", "t0": now - 7200, "t": now, "usd_per_h": 40.0}]   # later beat wins
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    usd, hours = R.spent(ledger)
    assert usd == pytest.approx(40 * (600 + 7200) / 3600) and hours == pytest.approx((600 + 7200) / 3600)
    assert R.spent(ledger, "train") == pytest.approx((80.0, 2.0))


def test_the_meter_stops_at_the_budget(env, monkeypatch):
    tmp, *_ = env
    monkeypatch.setattr(R, "gpu_info", lambda: ("NVIDIA H200", 8))
    c = cfg()
    c["budget"] = {"usd": 1.0, "hours": 0}
    stopped = []
    with R.Meter(c, "train", 32, 256, limits=True, every_s=0.05) as m:
        m.on_limit = lambda: stopped.append(True)
        m.row["t0"] -= 3600                      # pretend this attempt has run for an hour (about 40 USD)
        deadline = time.time() + 5
        while not stopped and time.time() < deadline:
            time.sleep(0.05)
    assert stopped and m.reason.startswith("budget reached")
    c["budget"] = {"usd": 0, "hours": 0.5}
    assert R.Meter(c, "train", 32, 256).over().startswith("time budget reached")
    c["budget"] = {"usd": 0, "hours": 0}
    assert R.Meter(c, "train", 32, 256).over() is None


def test_pool_report_reads_the_zero_advantage_share():
    """A 64-variant pool the policy has nearly solved gives GRPO almost nothing: D1 discarded 93% then 98% of groups."""
    def rows(rates, n=8):
        out = []
        for t, p in rates.items():
            s = round(p * n)
            out += [{"task": t, "valid": True, "ok": True, "reward": 1.0} for _ in range(s)]
            out += [{"task": t, "valid": True, "ok": True, "reward": 0.0} for _ in range(n - s)]
        return out
    fresh = R.pool_report(rows({f"v{i}": 0.5 for i in range(64)}), 8)
    assert fresh["mixed"] == 64 and fresh["expected_zero_advantage_share"] == pytest.approx(2 * 0.5 ** 8, abs=1e-3)
    late = R.pool_report(rows({**{f"v{i}": 1.0 for i in range(60)}, **{f"w{i}": 0.875 for i in range(4)}}), 8)
    assert late["ceiling"] == 60 and late["expected_zero_advantage_share"] > 0.9
    rep = R.pool_report([{"task": "a", "ok": False, "episode_failed": "create"}, {"task": "a", "valid": False, "dropped": "x"}], 8)
    assert rep["per_task"][0]["errored"] == 1 and rep["per_task"][0]["invalid"] == 1 and rep["scored"] == 0


def test_cli_render_writes_four_valid_tomls(env, capsys):
    tmp, split, ids = env
    assert R.main(["--config", str(TEMPLATES / "rl-config.toml"), "render", "--out", str(tmp / "out"), "--tasks", str(split)]) == 0
    for name in ("train", "smoke", "eval-inference", "eval"):
        tomllib.loads((tmp / "out" / f"{name}.toml").read_text())


def test_modal_app_imports_when_modal_is_installed():
    pytest.importorskip("modal")
    sys.path.insert(0, str(TEMPLATES / "modal"))
    try:
        import app  # noqa: F401
    finally:
        sys.path.pop(0)


def test_gpu_box_script_passes_bash_syntax():
    import subprocess
    assert subprocess.run(["bash", "-n", str(TEMPLATES / "gpu-box" / "run.sh")]).returncode == 0
    text = (TEMPLATES / "gpu-box" / "run.sh").read_text()
    assert "-e DRUMS_TOKEN)" in text and "DRUMS_TOKEN=" not in text.replace("${DRUMS_TOKEN:-}", "")


def test_harness_check_refuses_keys_the_installed_package_does_not_know(env, monkeypatch, tmp_path):
    """prime-rl's dry run accepts unknown harness keys (seen: drums-verifiers 0.2.2 with the window keys), so drums_rl
    checks them against the installed DrumsChatConfig before a run."""
    fake = tmp_path / "prime" / ".venv" / "bin"
    fake.mkdir(parents=True)
    (fake / "python").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    (fake / "python").chmod(0o755)
    monkeypatch.setattr(R, "PRIME_RL", tmp_path / "prime")
    from drums_chat import DrumsChatConfig
    tmp, split, ids = env
    text = R.render_train(cfg(), ids, tmp)
    if "screenshots_full" in DrumsChatConfig.model_fields:
        R.check_harness(text)
    with pytest.raises(SystemExit):
        R.check_harness(text.replace("env.agent.harness.downscale = 0.5", "env.agent.harness.no_such_key = 1"))


def test_train_resumes_past_a_partial_step_and_a_budget_stop_is_not_a_failure(env, monkeypatch):
    tmp, split, ids = env
    runs = tmp / "runs"
    (runs / "tasks" / "train").mkdir(parents=True)
    (runs / "tasks" / "train" / "tasks.jsonl").write_text((split / "tasks.jsonl").read_text())
    run = runs / "train"
    make_step(run, 1, True)
    make_step(run, 2, False)
    calls = []
    monkeypatch.setattr(R, "gpu_info", lambda: ("NVIDIA H200", 8))
    monkeypatch.setattr(R, "check_harness", lambda text: None)

    def fake_run(cmd, meter):
        calls.append(cmd)
        meter.reason = "budget reached: 501.00 of 500.00 USD"       # as the meter thread sets it
        return -15
    monkeypatch.setattr(R, "run_stoppable", fake_run)
    a = type("A", (), {"extra": ["--ckpt.interval", "2"]})()
    assert R.cmd_train(cfg(), a) == 0
    assert calls[0][:4] == ["uv", "run", "--no-sync", "rl"] and "--resume" in calls[0] and calls[0][-2:] == ["--ckpt.interval", "2"]
    assert not (run / "checkpoints" / "step_2").exists() and any((run / "checkpoints-partial").iterdir())
    stopped = json.loads((run / "STOPPED").read_text())
    assert stopped["reason"].startswith("budget reached") and stopped["last"].endswith("step_1")
    c = cfg()
    with open(runs / "spend.jsonl", "a") as f:                     # an earlier hour of 8 x H200
        f.write(json.dumps({"attempt": "train-0", "what": "train", "t0": 0, "t": 3600, "usd_per_h": 39.88}) + "\n")
    c["budget"] = {"usd": 30, "hours": 0}
    calls.clear()
    assert R.cmd_train(c, a) == 0 and calls == []                  # over budget: refuses to start
