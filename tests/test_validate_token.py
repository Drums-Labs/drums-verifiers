"""drums_tasks.validate finds its token the way the SDK and the docs say (A-14): $DRUMS_TOKEN by default, a file only when
asked for or as the fallback."""
import pytest

from drums_tasks.validate import resolve_token


def test_env_var_is_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".drums-token").write_text("from-legacy-file\n")
    assert resolve_token(None, {"DRUMS_TOKEN": "from-env"}) == "from-env"


def test_explicit_file_wins_over_the_env_var(tmp_path):
    f = tmp_path / "tok"
    f.write_text("from-file\n")
    assert resolve_token(str(f), {"DRUMS_TOKEN": "from-env"}) == "from-file"


def test_token_file_env_then_legacy_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    f = tmp_path / "tok"
    f.write_text("from-token-file-env\n")
    assert resolve_token(None, {"DRUMS_TOKEN_FILE": str(f)}) == "from-token-file-env"
    (tmp_path / ".drums-token").write_text("from-legacy-file\n")
    assert resolve_token(None, {}) == "from-legacy-file"


def test_no_token_anywhere_says_how_to_give_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="DRUMS_TOKEN"):
        resolve_token(None, {})
