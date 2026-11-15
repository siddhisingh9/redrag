"""Tests for OllamaClient using httpx.MockTransport (no Ollama server needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from redrag.llm import LLMError, OllamaClient

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
]


def make_client(handler, model: str = "llama3.2:3b") -> OllamaClient:
    return OllamaClient(model=model, transport=httpx.MockTransport(handler))


def test_generate_sends_expected_payload_and_returns_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello!"}})

    reply = make_client(handler).generate(MESSAGES, max_tokens=50, seed=1, temperature=0.0)

    assert reply == "Hello!"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "llama3.2:3b"
    assert seen["body"]["messages"] == MESSAGES
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"] == {"temperature": 0.0, "num_predict": 50, "seed": 1}


def test_default_temperature_used_when_not_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    make_client(handler).generate(MESSAGES)
    assert seen["body"]["options"] == {"temperature": 0.2}


def test_missing_model_gives_pull_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(LLMError, match="ollama pull llama3.2:3b"):
        make_client(handler).generate(MESSAGES)


def test_server_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError, match="HTTP 500"):
        make_client(handler).generate(MESSAGES)


def test_malformed_response_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(LLMError, match="Unexpected response"):
        make_client(handler).generate(MESSAGES)


def test_unreachable_server_raises_llm_error_and_is_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(LLMError, match="Is it running"):
        client.generate(MESSAGES)
    assert client.is_available() is False


def test_list_and_has_model():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "llama3.2:3b"}, {"name": "phi3:latest"}]}
        )

    assert make_client(handler).list_models() == ["llama3.2:3b", "phi3:latest"]
    assert make_client(handler).has_model() is True
    assert make_client(handler, model="phi3").has_model() is True  # "phi3" means "phi3:latest"
    assert make_client(handler, model="qwen2.5:3b").has_model() is False
