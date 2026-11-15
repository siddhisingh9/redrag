"""Tests for loading config from YAML + environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from redrag.config import PROJECT_ROOT, load_config, read_dotenv, resolve_path


def test_default_yaml_loads_with_expected_values():
    config = load_config(env={})
    assert config.llm.provider == "ollama"
    assert config.ollama.host == "http://localhost:11434"
    assert config.embeddings.model == "BAAI/bge-small-en-v1.5"
    assert config.rag.top_k == 4
    assert config.judge.unsure_low == 0.35
    assert config.judge.unsure_high == 0.65
    assert config.attacks.max_attempts == 5
    assert len(config.attacks.categories) == 5
    # Defenses are off by default: the target is deliberately vulnerable.
    assert not any(config.defenses.model_dump().values())
    # No key in YAML; it only ever comes from the environment.
    assert config.gemini.api_key is None


def test_env_overrides_yaml_and_converts_types():
    env = {
        "OLLAMA_HOST": "http://gpu-box:11434",
        "OLLAMA_VICTIM_MODEL": "qwen2.5:3b",
        "GEMINI_DAILY_BUDGET": "42",
        "REDRAG_LLM_PROVIDER": "mock",
    }
    config = load_config(env=env)
    assert config.ollama.host == "http://gpu-box:11434"
    assert config.ollama.victim_model == "qwen2.5:3b"
    assert config.gemini.daily_request_budget == 42  # "42" became an int
    assert config.llm.provider == "mock"


def test_empty_env_value_does_not_wipe_default():
    config = load_config(env={"GEMINI_MODEL": ""})
    assert config.gemini.model == "gemini-2.5-flash-lite"


def test_api_key_is_hidden_when_printed():
    config = load_config(env={"GEMINI_API_KEY": "super-secret-key-123"})
    assert config.gemini.api_key.get_secret_value() == "super-secret-key-123"
    assert "super-secret-key-123" not in repr(config)
    assert "super-secret-key-123" not in str(config.model_dump(mode="json"))


def test_invalid_provider_is_rejected():
    with pytest.raises(ValidationError):
        load_config(env={"REDRAG_LLM_PROVIDER": "openai"})


def test_unknown_yaml_key_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("rag:\n  topk: 4\n", encoding="utf-8")  # typo: topk
    with pytest.raises(ValidationError):
        load_config(path=bad, env={})


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(path=tmp_path / "nope.yaml", env={})


def test_redrag_config_env_selects_file(tmp_path: Path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("rag:\n  top_k: 9\n", encoding="utf-8")
    config = load_config(env={"REDRAG_CONFIG": str(custom)})
    assert config.rag.top_k == 9


def test_read_dotenv_parses_comments_quotes_and_blanks(tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        '# a comment\n\nGEMINI_MODEL=gemini-x\nOLLAMA_HOST="http://a:1"\nEMPTY=\nnot a line\n',
        encoding="utf-8",
    )
    assert read_dotenv(dotenv) == {
        "GEMINI_MODEL": "gemini-x",
        "OLLAMA_HOST": "http://a:1",
        "EMPTY": "",
    }


def test_read_dotenv_missing_file_returns_empty(tmp_path: Path):
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_resolve_path_is_relative_to_project_root(tmp_path: Path):
    assert resolve_path("runs") == PROJECT_ROOT / "runs"
    assert resolve_path(tmp_path) == tmp_path
