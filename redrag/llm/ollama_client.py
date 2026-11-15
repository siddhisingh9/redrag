"""Client for local models served by Ollama (https://ollama.com).

Ollama runs models on your own machine and exposes a small HTTP API. We use its
`/api/chat` endpoint directly with `httpx`, so no extra SDK is needed.

Ollama is used for two roles:
- the *victim* model inside the target RAG chatbot, and
- the *attacker* model that rewrites attack prompts (M3).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from redrag.llm.base import LLMError, Message, validate_messages

logger = logging.getLogger(__name__)


class OllamaClient:
    """Call one Ollama model through the `/api/chat` HTTP endpoint."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        temperature: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            model: Ollama model tag, e.g. "llama3.2:3b".
            host: Base URL of the Ollama server.
            timeout_seconds: Small models on a laptop GPU can be slow on the first
                call (the model is loaded into VRAM), so the default is generous.
            temperature: Default sampling temperature; can be overridden per call.
            transport: Optional httpx transport. Tests pass `httpx.MockTransport`
                so no real server is needed.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self._http = httpx.Client(base_url=self.host, timeout=timeout_seconds, transport=transport)

    def generate(self, messages: list[Message], **opts: Any) -> str:
        """Send the conversation to Ollama and return the reply text."""
        validate_messages(messages)

        # Ollama puts sampling settings under "options"; it calls the
        # max-length setting `num_predict` rather than `max_tokens`.
        options: dict[str, Any] = {"temperature": opts.get("temperature", self.temperature)}
        if "max_tokens" in opts:
            options["num_predict"] = opts["max_tokens"]
        if "seed" in opts:
            options["seed"] = opts["seed"]

        payload = {
            "model": self.model,
            "messages": messages,
            # stream=False returns one JSON object instead of a stream of pieces,
            # which is simpler to handle.
            "stream": False,
            "options": options,
        }

        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.host}. Is it running? ({exc})"
            ) from exc

        if response.status_code == 404:
            # Ollama answers 404 when the model has not been pulled yet.
            raise LLMError(f"Ollama model {self.model!r} not found. Run: ollama pull {self.model}")
        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")

        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected response from Ollama: {response.text[:300]}") from exc

    def list_models(self) -> list[str]:
        """Return the tags of models already pulled into Ollama."""
        try:
            response = self._http.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not list Ollama models at {self.host}: {exc}") from exc
        return [entry["name"] for entry in response.json().get("models", [])]

    def is_available(self) -> bool:
        """True if the Ollama server answers. Used to skip cleanly when it's off."""
        try:
            self.list_models()
        except LLMError:
            return False
        return True

    def has_model(self) -> bool:
        """True if this client's model has been pulled.

        Ollama lists "llama3.2:3b" but also accepts "llama3.2" meaning
        "llama3.2:latest", so we compare with and without the ":latest" tag.
        """
        wanted = {self.model, f"{self.model}:latest"}
        return any(name in wanted for name in self.list_models())
