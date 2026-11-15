"""The common interface every LLM client in RedRAG follows.

Why a shared interface? The target app, the attacker agent and the judge only
ever call `client.generate(messages)`. They don't care whether the answer comes
from Ollama on your GPU, Gemini in the cloud, or the scripted `MockClient` used
in tests. That is what lets every test run offline: we just swap in the mock.

Messages use the familiar chat format:

    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the library hours?"},
        {"role": "assistant", "content": "..."},
    ]
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]

VALID_ROLES = ("system", "user", "assistant")


class LLMError(RuntimeError):
    """Something went wrong talking to a model (network, bad response, ...)."""


class RateLimitError(LLMError):
    """The provider kept answering "too many requests" even after retrying."""


class BudgetExceededError(LLMError):
    """We refused to send a request because today's request budget is used up.

    This is raised *before* calling the API, so it never costs quota.
    """


@runtime_checkable
class LLMClient(Protocol):
    """Anything with this `generate` method can be used as an LLM in RedRAG.

    Supported `opts` (each client maps them to its own API):
        temperature (float): randomness; lower is more repeatable.
        max_tokens (int): maximum length of the reply.
        seed (int): fixed seed for repeatable sampling, where supported.
    """

    model: str

    def generate(self, messages: list[Message], **opts: Any) -> str:
        """Send a chat conversation and return the model's reply text."""
        ...


def validate_messages(messages: list[Message]) -> None:
    """Check that `messages` is a non-empty list of {role, content} dicts.

    Catching a malformed message here gives a clear error, instead of a vague
    "400 Bad Request" from a remote API later.
    """
    if not messages:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be a dict, got {type(message).__name__}")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(
                f"messages[{index}] has invalid role {role!r}; use one of {VALID_ROLES}"
            )
        if not isinstance(message.get("content"), str):
            raise ValueError(f"messages[{index}] must have a string 'content'")
