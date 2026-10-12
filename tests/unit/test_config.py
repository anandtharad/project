from __future__ import annotations

from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from app.config import Settings, detect_env_file_encoding
from tests.helpers import make_settings, suite_of


def test_defaults_are_valid() -> None:
    settings = Settings(_env_file=None)
    assert settings.generation_provider == "anthropic"
    assert settings.generation_model is None  # resolved from the provider preset
    assert settings.stale_turn_seconds > settings.turn_deadline_seconds


def test_production_refuses_dev_auth() -> None:
    with pytest.raises(ValidationError, match="production authentication is not implemented"):
        make_settings(app_env="production", generation_provider="anthropic")


def test_stale_turn_must_exceed_deadline() -> None:
    with pytest.raises(ValidationError, match="STALE_TURN_SECONDS"):
        make_settings(turn_deadline_seconds=60, stale_turn_seconds=30)


def test_request_byte_limit_must_cover_character_limit() -> None:
    with pytest.raises(ValidationError, match="MAX_REQUEST_BYTES"):
        make_settings(max_input_characters=20_000, max_request_bytes=2048)


def test_dev_token_parsing() -> None:
    settings = make_settings(dev_auth_tokens=" alice:tok-a , bob:tok-b ,")
    assert settings.dev_token_map() == {"tok-a": "alice", "tok-b": "bob"}
    assert make_settings(dev_auth_tokens=None).dev_token_map() == {}


def test_malformed_dev_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="handle:token"):
        make_settings(dev_auth_tokens="alice").dev_token_map()


def test_secrets_are_not_in_repr() -> None:
    settings = make_settings(generation_api_key="sk-ant-secret-value")
    assert "sk-ant-secret-value" not in repr(settings)
    assert "token-alice" not in repr(settings)


@pytest.mark.parametrize(
    ("raw", "encoding"),
    [
        # PowerShell 5.1 `echo ... > .env` writes UTF-16 LE with a BOM.
        ("GROQ_API_KEY=abc\r\nGENERATION_PROVIDER=groq\r\n".encode("utf-16"), "utf-16"),
        # Notepad / Set-Content -Encoding utf8 may add a UTF-8 BOM.
        (b"\xef\xbb\xbfGROQ_API_KEY=abc\r\nGENERATION_PROVIDER=groq\r\n", "utf-8-sig"),
        (b"GROQ_API_KEY=abc\nGENERATION_PROVIDER=groq\n", "utf-8-sig"),
    ],
)
def test_env_file_written_by_windows_tools_is_read(
    tmp_path: Path, raw: bytes, encoding: str
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(raw)
    assert detect_env_file_encoding(env_file) == encoding
    settings = Settings(_env_file=env_file)
    assert settings.generation_provider == "groq"
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "abc"


def test_missing_env_file_is_fine(tmp_path: Path) -> None:
    assert detect_env_file_encoding(tmp_path / "absent.env") is None
    assert detect_env_file_encoding(None) is None


@pytest.mark.parametrize(
    ("path", "suite"),
    [
        (PureWindowsPath(r"C:\anand\project\tests\live\test_groq_live.py"), "live"),
        (PureWindowsPath(r"C:\anand\project\tests\integration\test_turns.py"), "integration"),
        (PurePosixPath("/home/me/project/tests/security/test_input.py"), "security"),
        (PurePosixPath("/home/me/project/tests/conftest.py"), None),
    ],
)
def test_suite_markers_work_with_windows_and_posix_paths(path: PurePath, suite: str | None) -> None:
    assert suite_of(path) == suite
