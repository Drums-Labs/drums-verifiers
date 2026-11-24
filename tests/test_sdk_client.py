"""The client this package builds leaves 429 admission to DrumsEnv._admitted: with drums-sdk 0.4.0 or later the SDK's
own 429 retry is off (RetryPolicy(max_wait_s=0)); with 0.3.0, which has no RetryPolicy, connect() is called as before."""
import types

from drums_verifiers import sdk


def _fake_drums(with_policy: bool):
    calls = []

    class RetryPolicy:
        def __init__(self, max_wait_s=60.0):
            self.max_wait_s = max_wait_s

    def connect(url, token, **kw):
        calls.append((url, token, kw))
        return object()

    mod = types.SimpleNamespace(connect=connect)
    if with_policy:
        mod.RetryPolicy = RetryPolicy
    return mod, calls


def test_sdk_retry_is_off_when_the_sdk_has_a_retry_policy(monkeypatch):
    mod, calls = _fake_drums(with_policy=True)
    monkeypatch.setitem(__import__("sys").modules, "drums", mod)
    monkeypatch.setenv("DRUMS_TOKEN", "t")
    monkeypatch.delenv("DRUMS_TOKEN_FILE", raising=False)
    sdk.client.cache_clear()
    sdk.client("http://127.0.0.1:1")
    sdk.client.cache_clear()
    (_, _, kw), = calls
    assert kw["retry"].max_wait_s == 0


def test_an_sdk_without_retry_policy_is_called_as_before(monkeypatch):
    mod, calls = _fake_drums(with_policy=False)
    monkeypatch.setitem(__import__("sys").modules, "drums", mod)
    monkeypatch.setenv("DRUMS_TOKEN", "t")
    monkeypatch.delenv("DRUMS_TOKEN_FILE", raising=False)
    sdk.client.cache_clear()
    sdk.client("http://127.0.0.1:1")
    sdk.client.cache_clear()
    assert calls == [("http://127.0.0.1:1", "t", {})]


def test_the_installed_sdk_accepts_the_policy():
    import drums
    if getattr(drums, "RetryPolicy", None) is None:
        assert sdk.no_sdk_admission_retry() is None
    else:
        assert sdk.no_sdk_admission_retry().max_wait_s == 0
