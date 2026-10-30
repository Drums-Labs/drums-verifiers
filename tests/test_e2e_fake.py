"""The whole verifiers path (vf-eval -> DrumsEnv -> null harness -> computer_* MCP tools -> drums SDK) against a fake drumsd
and the scripted mock policy. Checks: valid episodes get the checker's score; a fork group's children are scored one by
one; children whose verdict is invalid are DROPPED with no reward even when the checker said 1.0."""
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VF_EVAL = shutil.which("vf-eval") or os.path.join(os.path.dirname(sys.executable), "vf-eval")
pytestmark = pytest.mark.skipif(not os.path.exists(VF_EVAL), reason="vf-eval not installed")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    (d / "tok").write_text("t0k\n")
    dp, mp = free_port(), free_port()
    cp = free_port()
    mock = [sys.executable, os.path.join(ROOT, "examples", "mock_policy.py")]
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "fake_drumsd.py"), str(dp), "t0k"]),
             subprocess.Popen(mock + ["--port", str(mp)]), subprocess.Popen(mock + ["--port", str(cp), "--plans", "crash"])]
    time.sleep(1.5)
    yield d, dp, mp, cp
    for p in procs:
        p.kill()


def run_eval(d, dp, mp, cp, name, *extra, crash=False):
    mp = cp if crash else mp
    rec = d / f"{name}.jsonl"
    env = dict(os.environ, DRUMS_URL=f"http://127.0.0.1:{dp}", DRUMS_TOKEN_FILE=str(d / "tok"), DRUMS_RECORD=str(rec), MOCK_KEY="x")
    r = subprocess.run([VF_EVAL, "drums-verifiers", "-m", "mock", "-n", "1", "--no-push", "--client.base-url",
                        f"http://127.0.0.1:{mp}/v1", "--client.api-key-var", "MOCK_KEY", "-o", str(d / name), *extra],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return [json.loads(l) for l in open(rec)]


def test_single_env_rollouts_get_the_checker_score(servers):
    rows = run_eval(*servers, "single", "-r", "2")
    assert len(rows) == 2 and all(r["valid"] and r["ok"] and r["dropped"] is None for r in rows)
    assert sorted(r["reward"] for r in rows) == [0.0, 1.0]           # one probe, one scripted solve


def test_fork_group_children_are_scored_each(servers):
    rows = run_eval(*servers, "fork", "--env.fork-n", "4", "--env.prefix", '["pyautogui.moveTo(10, 10)"]', "--env.settle-s", "0")
    assert len(rows) == 4 and len({r["parent"] for r in rows}) == 1 and len({r["env"] for r in rows}) == 4
    assert sorted(r["reward"] for r in rows) == [0.0, 0.0, 1.0, 1.0]


def test_invalid_children_are_dropped_never_scored(servers):
    rows = run_eval(*servers, "invalid", "--env.fork-n", "4", "--env.prefix", '["KILL_CHILDREN = 1"]', "--env.settle-s", "0")
    assert len(rows) == 4
    for r in rows:
        assert r["valid"] is False and r["reward"] is None and r["rewards"] == {"task_success": None}
        assert r["dropped"].startswith("drums_invalid:invalid:alive_after_resume")
    assert any(r["score"] == 1.0 for r in rows)                       # the checker said 1.0; it never became a reward


def test_env_failure_mid_episode_is_an_error_not_a_reward(servers):
    rows = run_eval(*servers, "crash", crash=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["ok"] is False and r["reward"] is None and r["valid"] is None
    assert any("Drums env failed" in e for e in r["errors"]) and r["stop"] == "env_failed"


def test_reuse_guard_refuses_an_env_that_was_already_acted_on(servers, monkeypatch):
    import asyncio

    import drums
    from drums_verifiers import sdk
    from drums_verifiers.taskset import DrumsTask, DrumsTaskData, step_count
    d, dp, *_ = servers
    monkeypatch.setenv("DRUMS_URL", f"http://127.0.0.1:{dp}")
    monkeypatch.setenv("DRUMS_TOKEN_FILE", str(d / "tok"))
    sdk.client.cache_clear()
    c = sdk.client()
    env = c.create("osworld-ubuntu", c.task("e0df059f"))
    data = DrumsTaskData(prompt=None, osworld={"id": "x"}, drums_env_id=env.id, drums_episode=1, drums_steps=step_count(env))
    asyncio.run(DrumsTask(data).setup(trace=None))                     # untouched: accepted
    env.step("pyautogui.press('a')")
    with pytest.raises(RuntimeError, match="earlier attempt"):
        asyncio.run(DrumsTask(data).setup(trace=None))
    assert isinstance(env, drums.Env)
