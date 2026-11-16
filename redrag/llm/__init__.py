"""LLM clients (Ollama, Gemini, Mock) behind one shared `LLMClient` interface.

Use the `build_*` helpers instead of constructing clients by hand, so the
choice between real and mock models stays in one place (the config).
"""

from __future__ import annotations

from typing import Literal

from redrag.config import Config, resolve_path
from redrag.llm.base import (
    BudgetExceededError,
    LLMClient,
    LLMError,
    Message,
    RateLimitError,
)
from redrag.llm.gemini_client import GeminiClient
from redrag.llm.mock_client import MockClient
from redrag.llm.ollama_client import OllamaClient

__all__ = [
    "BudgetExceededError",
    "GeminiClient",
    "LLMClient",
    "LLMError",
    "Message",
    "MockClient",
    "OllamaClient",
    "RateLimitError",
    "build_gemini_client",
    "build_local_client",
]


def build_local_client(config: Config, role: Literal["victim", "attacker"]) -> LLMClient:
    """Build the local model for the target app ("victim") or the attacker agent.

    `llm.provider: mock` (or `REDRAG_LLM_PROVIDER=mock`) returns the offline
    MockClient, which is what tests and the CI workflow use.
    """
    if config.llm.provider == "mock":
        return MockClient(model=f"mock-{role}")

    model = config.ollama.victim_model if role == "victim" else config.ollama.attacker_model
    return OllamaClient(
        model=model,
        host=config.ollama.host,
        timeout_seconds=config.ollama.timeout_seconds,
        temperature=config.ollama.temperature,
    )


def build_gemini_client(config: Config) -> GeminiClient:
    """Build the Gemini client (the backup judge) from config."""
    gemini = config.gemini
    api_key = gemini.api_key.get_secret_value() if gemini.api_key else None
    return GeminiClient(
        model=gemini.model,
        api_key=api_key,
        cache_dir=resolve_path(gemini.cache_dir),
        daily_request_budget=gemini.daily_request_budget,
        min_seconds_between_requests=gemini.min_seconds_between_requests,
        max_retries=gemini.max_retries,
        backoff_base_seconds=gemini.backoff_base_seconds,
    )
