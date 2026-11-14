"""The whole verifiers path (vf-run -> DrumsEnv -> null harness -> computer_* MCP tools -> drums SDK) against a fake drumsd
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
VF_RUN = shutil.which("vf-run") or os.path.join(os.path.dirname(sys.executable), "vf-run")
pytestmark = pytest.mark.skipif(not os.path.exists(VF_RUN), reason="vf-run not installed")


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
    cp, qp, lp = free_port(), free_port(), free_port()
    mock = [sys.executable, os.path.join(ROOT, "examples", "mock_policy.py")]
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "fake_drumsd.py"), str(dp), "t0k"]),
             subprocess.Popen(mock + ["--port", str(mp)]), subprocess.Popen(mock + ["--port", str(cp), "--plans", "crash"]),
             subprocess.Popen(mock + ["--port", str(qp), "--plans", "quarantine"]),
             subprocess.Popen(mock + ["--port", str(lp), "--plans", "long"])]
    time.sleep(1.5)
    yield d, dp, mp, (cp, qp, lp)
    for p in procs:
        p.kill()


def run_eval(d, dp, mp, cpq, name, *extra, crash=False, quarantine=False, long=False):
    mp = cpq[0] if crash else cpq[1] if quarantine else cpq[2] if long else mp
    rec = d / f"{name}.jsonl"
    env = dict(os.environ, DRUMS_URL=f"http://127.0.0.1:{dp}", DRUMS_TOKEN_FILE=str(d / "tok"), DRUMS_RECORD=str(rec), MOCK_KEY="x")
    r = subprocess.run([VF_RUN, "drums-verifiers", "-m", "mock", "-n", "1", "--client.base-url",
                        f"http://127.0.0.1:{mp}/v1", "--client.api-key-var", "MOCK_KEY", "-o", str(d / name), *extra],
                       env=env, capture_output=True, text=True, timeout=600)
    # vf-run exits non-zero when every episode errored (the crash scenario); the record is still what we check
    assert r.returncode == 0 or crash, r.stdout[-2000:] + r.stderr[-2000:]
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


def test_quarantine_mid_episode_is_dropped_not_an_error(servers):
    rows = run_eval(*servers, "quarantine", quarantine=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["ok"] is True and r["reward"] is None and r["valid"] is False and r["stop"] == "env_failed"
    assert r["dropped"].startswith("drums_invalid:invalid:alive_after_resume")


def test_scalecua_tool_style_end_to_end(servers):
    rows = run_eval(*servers, "scalecua", "-r", "2", "--env.taskset.task.tools.style", "scalecua")
    assert len(rows) == 2 and all(r["valid"] and r["ok"] for r in rows)
    assert sorted(r["reward"] for r in rows) == [0.0, 1.0]


def test_child_prefix_control_children_dropped(servers):
    rows = run_eval(*servers, "childprefix", "--env.fork-n", "3", "--env.prefix", '["pyautogui.moveTo(1, 1)"]',
                    "--env.child-prefix", '["QUARANTINE = 1"]', "--env.settle-s", "0")
    assert len(rows) == 3, rows
    assert all(r["reward"] is None and (r["dropped"] or "").startswith("drums_invalid") for r in rows), [(r["ok"], r["dropped"], r["errors"], r.get("child_prefix")) for r in rows]


def test_turn_cap_never_runs_an_unrecorded_action(servers):
    """At max_turns the last allowed response's tool call must not run: every executed action has its tool result
    (and screenshot) in the trace, and drumsd ran exactly those actions."""
    import glob
    d = servers[0]
    rows = run_eval(*servers, "turncap", "--env.agent.max-turns", "6", "--env.taskset.task.tools.pause-s", "0", long=True)
    r = rows[0]
    assert r["stop"] == "turn_cap" and r["turns"] == 6 and r["actions"] == 5, r
    ep = json.loads(open(glob.glob(str(d / "turncap" / "*" / "traces.jsonl"))[0]).readline())
    t = ep["traces"][0]
    results = {n["message"].get("tool_call_id") for n in t["nodes"] if n["message"].get("role") == "tool"
               and any(p.get("type") == "image_url" for p in (n["message"].get("content") or []) if isinstance(p, dict))}
    assert len(results) == r["actions"] == 5


def test_env_that_never_starts_is_recorded_as_a_failed_episode(servers):
    """A restore that crashes before any desktop exists (no agent, no trace) must still leave a record row, so the episode
    is counted as failed instead of silently missing (found in the soak: one crashed create left no row)."""
    d, dp, mp, cpq = servers
    rec = d / "nostart.jsonl"
    env = dict(os.environ, DRUMS_URL=f"http://127.0.0.1:{dp}", DRUMS_TOKEN_FILE=str(d / "tok"), DRUMS_RECORD=str(rec), MOCK_KEY="x")
    subprocess.run([VF_RUN, "drums-verifiers", "-m", "mock", "-n", "1", "--client.base-url", f"http://127.0.0.1:{mp}/v1",
                    "--client.api-key-var", "MOCK_KEY", "-o", str(d / "nostart"), "--env.taskset.task-ids",
                    '["fa11ed00-0000-4000-8000-000000000000"]'], env=env, capture_output=True, text=True, timeout=600)
    rows = [json.loads(l) for l in open(rec)]
    assert len(rows) == 1 and rows[0]["env_create_failed"] and rows[0]["ok"] is False and rows[0]["reward"] is None
    assert "restore failed" in rows[0]["error"]
    from drums_verifiers import sdk                                 # the down env is closed, not left in the live list
    sdk.client.cache_clear()
    rec_env = sdk.env(rows[0]["env"], f"http://127.0.0.1:{dp}", str(d / "tok")).record
    assert rows[0]["env"] and rec_env["state"] == "destroyed", rec_env


def test_fork_refused_by_admission_waits_and_retries(servers):
    """drumsd answers 429 Overloaded (memory admission; nothing started) to a fork: the episode waits and asks again, and
    the group still plays (found in the fresh-host mock acceptance: 4 concurrent fork episodes on a c8i.4xlarge, one fork
    refused, the episode lost with no record row)."""
    rows = run_eval(*servers, "overload", "--env.fork-n", "2", "--env.prefix", '["OVERLOAD_TWICE = 1"]', "--env.settle-s", "0")
    assert len(rows) == 2 and all(r["valid"] and r["ok"] for r in rows)
    assert [w["reason"] for w in rows[0]["admission_waits"]] == ["memory", "memory"]


def test_fork_refused_past_the_wait_is_a_recorded_failed_episode(servers):
    d, dp, mp, cpq = servers
    rec = d / "overload_always.jsonl"
    env = dict(os.environ, DRUMS_URL=f"http://127.0.0.1:{dp}", DRUMS_TOKEN_FILE=str(d / "tok"), DRUMS_RECORD=str(rec), MOCK_KEY="x")
    subprocess.run([VF_RUN, "drums-verifiers", "-m", "mock", "-n", "1", "--client.base-url", f"http://127.0.0.1:{mp}/v1",
                    "--client.api-key-var", "MOCK_KEY", "-o", str(d / "overload_always"), "--env.fork-n", "2", "--env.prefix",
                    '["OVERLOAD_ALWAYS = 1"]', "--env.settle-s", "0", "--env.admission-wait-s", "3"],
                   env=env, capture_output=True, text=True, timeout=600)
    rows = [json.loads(l) for l in open(rec)]
    assert len(rows) == 1 and rows[0]["episode_failed"] == "fork" and rows[0]["ok"] is False and rows[0]["reward"] is None
    assert "Overloaded" in rows[0]["error"] and len(rows[0]["admission_waits"]) >= 1


def test_close_during_a_drumsd_restart_is_retried(servers):
    """A DELETE that finds drumsd restarting (503) is asked again, so no env is left behind (found in the real-model soak:
    a fork group's closes fell in the 1.28 s drumsd restart after the injected SIGKILL; 3 slots and the fork state leaked)."""
    rows = run_eval(*servers, "flakydelete", "--env.fork-n", "2", "--env.prefix", '["FLAKY_DELETE = 1"]', "--env.settle-s", "0")
    assert len(rows) == 2 and not any(r.get("cleanup_failed") for r in rows)
    from drums_verifiers import sdk
    d, dp, *_ = servers
    sdk.client.cache_clear()
    for r in rows + [{"env": rows[0]["parent"]}]:
        assert sdk.env(r["env"], f"http://127.0.0.1:{dp}", str(d / "tok")).record["state"] == "destroyed", r["env"]
