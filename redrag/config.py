"""Load RedRAG settings from `config/default.yaml`, then apply environment overrides.

Why two sources?
- The YAML file holds sensible defaults that are safe to commit.
- Environment variables (or a local `.env` file) hold things that differ per machine
  or must stay secret, like `GEMINI_API_KEY`. They win over the YAML.

Every setting is validated by a pydantic model, so a typo such as `top_k: four`
fails loudly at startup instead of causing a confusing error deep inside a scan.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

# The project root is the folder that contains the `redrag/` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


class _Section(BaseModel):
    """Base for config sections: reject unknown keys so typos are caught early."""

    model_config = ConfigDict(extra="forbid")


class TargetConfig(_Section):
    url: str = "http://127.0.0.1:8000"
    host: str = "127.0.0.1"
    port: int = 8000


class LLMConfig(_Section):
    provider: Literal["ollama", "mock"] = "ollama"


class OllamaConfig(_Section):
    host: str = "http://localhost:11434"
    victim_model: str = "llama3.2:3b"
    attacker_model: str = "llama3.2:3b"
    timeout_seconds: float = 120.0
    temperature: float = 0.2


class GeminiConfig(_Section):
    model: str = "gemini-2.5-flash-lite"
    # SecretStr hides the value when the config is printed or logged.
    # The key never comes from YAML, only from the environment / .env.
    api_key: SecretStr | None = None
    daily_request_budget: int = Field(default=200, ge=0)
    min_seconds_between_requests: float = Field(default=4.0, ge=0)
    max_retries: int = Field(default=5, ge=0)
    backoff_base_seconds: float = Field(default=2.0, ge=0)
    cache_dir: str = ".cache/gemini"


class EmbeddingsConfig(_Section):
    # "sentence-transformers" = the real model below; "hashing" = a tiny offline
    # word-matching embedder used by tests and CI (no model download).
    provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"


class RAGConfig(_Section):
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    top_k: int = Field(default=4, gt=0)
    chroma_dir: str = "data/chroma"
    collection: str = "handbook"


class DefensesConfig(_Section):
    input_filter: bool = False
    context_sanitiser: bool = False
    output_leak_filter: bool = False
    judge_guard: bool = False


class JudgeConfig(_Section):
    classifier_path: str = "models/judge-deberta"
    unsure_low: float = Field(default=0.35, ge=0, le=1)
    unsure_high: float = Field(default=0.65, ge=0, le=1)
    overlap_threshold: float = Field(default=0.5, ge=0, le=1)
    # Ask Gemini when the classifier is unsure (needs GEMINI_API_KEY).
    use_gemini: bool = True
    # judge_guard blocks an answer when the classifier's P(success) reaches this.
    guard_threshold: float = Field(default=0.65, ge=0, le=1)
    labels_file: str = "data/labels/hand_labels.jsonl"
    dataset_dir: str = "data/judge_dataset"


class AttacksConfig(_Section):
    categories: list[str] = Field(
        default_factory=lambda: [
            "direct_injection",
            "indirect_injection",
            "system_prompt_extraction",
            "data_exfiltration",
            "multi_turn_escalation",
        ]
    )
    max_attempts: int = Field(default=5, gt=0)


class OutputConfig(_Section):
    runs_dir: str = "runs"


class TracingConfig(_Section):
    # Off by default, so scans work without Phoenix running. Turn on with --trace.
    enabled: bool = False
    # Phoenix's OpenTelemetry endpoint (start Phoenix with `phoenix serve`).
    endpoint: str = "http://localhost:6006/v1/traces"
    project: str = "redrag"


class Config(_Section):
    """The full RedRAG configuration. Each attribute matches a top-level YAML key."""

    target: TargetConfig = Field(default_factory=TargetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    defenses: DefensesConfig = Field(default_factory=DefensesConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    attacks: AttacksConfig = Field(default_factory=AttacksConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


# Which environment variable overrides which config value: (section, key).
# Kept as an explicit table (rather than a clever naming scheme) so it is easy to
# see every override in one place. `.env.example` documents each of these.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "REDRAG_TARGET_URL": ("target", "url"),
    "REDRAG_LLM_PROVIDER": ("llm", "provider"),
    "REDRAG_EMBEDDINGS_PROVIDER": ("embeddings", "provider"),
    "OLLAMA_HOST": ("ollama", "host"),
    "OLLAMA_VICTIM_MODEL": ("ollama", "victim_model"),
    "OLLAMA_ATTACKER_MODEL": ("ollama", "attacker_model"),
    "GEMINI_API_KEY": ("gemini", "api_key"),
    "GEMINI_MODEL": ("gemini", "model"),
    "GEMINI_DAILY_BUDGET": ("gemini", "daily_request_budget"),
    "REDRAG_JUDGE_MODEL_DIR": ("judge", "classifier_path"),
    "REDRAG_LABELS_FILE": ("judge", "labels_file"),
    "REDRAG_DATASET_DIR": ("judge", "dataset_dir"),
    "REDRAG_RUNS_DIR": ("output", "runs_dir"),
    "REDRAG_TRACING": ("tracing", "enabled"),
    "REDRAG_TRACING_ENDPOINT": ("tracing", "endpoint"),
}


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple `.env` file into a dict.

    Supports `KEY=value` lines, `#` comments, blank lines, and optional quotes
    around the value. That covers everything `.env.example` uses, so we don't
    need an extra dependency for it.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Strip one pair of matching surrounding quotes: KEY="a b" -> a b
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def build_environment(dotenv_path: Path | None = None) -> dict[str, str]:
    """Merge `.env` values with the real process environment.

    The real environment wins, so `OLLAMA_HOST=... redrag scan` works even if
    `.env` sets a different value. `dotenv_path` defaults to `DEFAULT_DOTENV_PATH`,
    looked up at call time so tests can point it at a file that doesn't exist
    (and never pick up your real API key).
    """
    merged: dict[str, str] = {}
    merged.update(read_dotenv(dotenv_path or DEFAULT_DOTENV_PATH))
    merged.update(os.environ)
    return merged


def load_config(
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load and validate the configuration.

    Args:
        path: YAML file to read. Defaults to `$REDRAG_CONFIG` or `config/default.yaml`.
        env: Environment variables to apply as overrides. Defaults to the real
            environment merged with `.env`. Tests pass an explicit dict so the
            developer's own `.env` (and API key) never leaks into test runs.
    """
    if env is None:
        env = build_environment()

    if path is None:
        path = env.get("REDRAG_CONFIG") or DEFAULT_CONFIG_PATH
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")

    # Apply overrides. Empty values (e.g. `GEMINI_API_KEY=` in .env) are skipped,
    # so an unfilled line doesn't wipe a default. Pydantic converts the strings
    # to the right type (e.g. "200" -> 200) during validation below.
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = env.get(env_name)
        if value:
            data.setdefault(section, {})[key] = value

    return Config.model_validate(data)


def resolve_path(path: Path | str) -> Path:
    """Turn a config path into an absolute path, relative to the project root.

    This makes `runs/` or `.cache/gemini` mean the same folder no matter which
    directory you run `redrag` from.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
