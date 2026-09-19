from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, load_settings
from app.errors import ConfigError


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "EMBEDDING_MODEL",
        "DOCUMENTS_PATH",
        "INDEX_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "SIMILARITY_THRESHOLD",
        "MAX_HISTORY_MESSAGES",
        "MAX_INPUT_CHARACTERS",
        "MODEL_TIMEOUT_SECONDS",
        "MODEL_MAX_TOKENS",
        "MODEL_MAX_RETRIES",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_defaults_load_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == ""
    assert settings.chunk_size == 800
    assert settings.top_k == 5


def test_require_groq_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigError):
        settings.require_groq_api_key()


def test_require_groq_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = Settings(_env_file=None, GROQ_API_KEY="secret-key")
    assert settings.require_groq_api_key() == "secret-key"


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=100, CHUNK_OVERLAP=100)


def test_similarity_threshold_out_of_range_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, SIMILARITY_THRESHOLD=1.5)


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOG_LEVEL="NOISY")


def test_load_settings_wraps_errors_as_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    with pytest.raises(ConfigError):
        load_settings()


def test_index_version_key_changes_with_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    a = Settings(_env_file=None, EMBEDDING_MODEL="model-a")
    b = Settings(_env_file=None, EMBEDDING_MODEL="model-b")
    assert a.index_version_key() != b.index_version_key()
