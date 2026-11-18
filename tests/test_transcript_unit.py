"""Offline unit tests of the transcript pieces (A-42): the status of a tool call and per-turn token counts."""
from types import SimpleNamespace

from drums_verifiers.taskset import turn_tokens
from drums_verifiers.tools import call_status


def test_call_status_from_the_state_counters():
    base = (0, None, None)
    assert call_status(base, base, ["click: done.", object()]) == "ok"
    assert call_status(base, (1, None, None), "Invalid action: x") == "invalid"
    assert call_status(base, (0, "click: boom", None), "The environment stopped while running click") == "env_failed"
    assert call_status(base, (0, None, "quarantined"), "...") == "quarantined"
    assert call_status((0, "x", None), (0, "x", None), "The environment stopped earlier (x); no action was taken.") == "skipped"
    assert call_status(base, base, "DONE: no action.") == "no_action"


def test_turn_tokens_with_and_without_usage():
    used = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=30, cached_input_tokens=70, completion_tokens=5),
                           time=SimpleNamespace(duration=1.23456))
    bare = SimpleNamespace(usage=None, time=SimpleNamespace(duration=0.5))
    out = turn_tokens(SimpleNamespace(calls=[used, bare]))
    assert out == [{"turn": 0, "prompt_tokens": 100, "cached_tokens": 70, "completion_tokens": 5, "s": 1.235},
                   {"turn": 1, "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None, "s": 0.5}]
    assert turn_tokens(SimpleNamespace()) == []
