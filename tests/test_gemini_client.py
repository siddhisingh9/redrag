"""Tests for GeminiClient: caching, daily budget, backoff on 429 and key safety.

A tiny fake SDK client stands in for `google.genai.Client`, and time is faked
(`sleep`, `monotonic`, `today` are injected), so these tests are instant and offline.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from redrag.llm import BudgetExceededError, GeminiClient, LLMError, RateLimitError
from redrag.llm.gemini_client import is_retryable_error, to_gemini_contents

MESSAGES = [
    {"role": "system", "content": "You are a strict judge."},
    {"role": "user", "content": "Did the attack succeed?"},
    {"role": "assistant", "content": "Let me check."},
    {"role": "user", "content": "Answer yes or no."},
]


class FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError, which carries the HTTP status in `.code`."""

    def __init__(self, code: int, message: str = "error") -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class FakeSDK:
    """Fake `genai.Client`: `fake.models.generate_content(...)` replays `outcomes`.

    Each outcome is either a reply string or an exception to raise.
    """

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.models = self  # so `client.models.generate_content` works

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


class FakeClock:
    """Records sleeps and advances a fake monotonic clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def make_client(tmp_path: Path, sdk: FakeSDK, clock: FakeClock | None = None, **kwargs):
    clock = clock or FakeClock()
    defaults = {
        "model": "gemini-test",
        "api_key": None,
        "cache_dir": tmp_path / "cache",
        "daily_request_budget": 10,
        "min_seconds_between_requests": 0.0,
        "max_retries": 3,
        "backoff_base_seconds": 1.0,
        "sdk_client": sdk,
        "sleep": clock.sleep,
        "monotonic": clock.monotonic,
        "today": lambda: date(2026, 11, 15),
    }
    defaults.update(kwargs)
    return GeminiClient(**defaults)


def test_messages_converted_to_gemini_format():
    system, contents = to_gemini_contents(MESSAGES)
    assert system == "You are a strict judge."
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"] == [{"text": "Did the attack succeed?"}]


def test_generate_passes_model_system_instruction_and_options(tmp_path):
    sdk = FakeSDK(["yes"])
    client = make_client(tmp_path, sdk)
    assert client.generate(MESSAGES, temperature=0, max_tokens=5) == "yes"

    request = sdk.requests[0]
    assert request["model"] == "gemini-test"
    assert request["config"] == {
        "system_instruction": "You are a strict judge.",
        "temperature": 0,
        "max_output_tokens": 5,
    }


def test_cache_hit_skips_api_and_budget(tmp_path):
    sdk = FakeSDK(["yes"])  # only one reply available: a 2nd API call would fail
    client = make_client(tmp_path, sdk)
    assert client.generate(MESSAGES) == "yes"
    assert client.generate(MESSAGES) == "yes"
    assert len(sdk.requests) == 1
    assert client.requests_used_today() == 1

    # A brand-new client (e.g. a re-run of the scan) also reads the disk cache.
    fresh = make_client(tmp_path, FakeSDK([]))
    assert fresh.generate(MESSAGES) == "yes"


def test_different_options_are_cached_separately(tmp_path):
    sdk = FakeSDK(["a", "b"])
    client = make_client(tmp_path, sdk)
    assert client.generate(MESSAGES, temperature=0) == "a"
    assert client.generate(MESSAGES, temperature=1) == "b"


def test_retries_429_with_exponential_backoff(tmp_path):
    clock = FakeClock()
    sdk = FakeSDK([FakeAPIError(429), FakeAPIError(429), "finally"])
    client = make_client(tmp_path, sdk, clock=clock, backoff_base_seconds=1.0)

    assert client.generate(MESSAGES) == "finally"
    assert len(sdk.requests) == 3
    # Waits ~1s then ~2s (plus up to 0.5s random jitter each).
    assert len(clock.sleeps) == 2
    assert 1.0 <= clock.sleeps[0] <= 1.5
    assert 2.0 <= clock.sleeps[1] <= 2.5
    # Only the successful call counts against the daily budget.
    assert client.requests_used_today() == 1


def test_gives_up_after_max_retries(tmp_path):
    sdk = FakeSDK([FakeAPIError(429)] * 4)
    client = make_client(tmp_path, sdk, max_retries=3)
    with pytest.raises(RateLimitError):
        client.generate(MESSAGES)
    assert len(sdk.requests) == 4  # first try + 3 retries
    assert client.requests_used_today() == 0


def test_non_retryable_error_fails_immediately(tmp_path):
    sdk = FakeSDK([FakeAPIError(400, "INVALID_ARGUMENT"), "never used"])
    client = make_client(tmp_path, sdk)
    with pytest.raises(LLMError, match="Gemini request failed"):
        client.generate(MESSAGES)
    assert len(sdk.requests) == 1


def test_budget_blocks_before_calling_api_and_persists(tmp_path):
    sdk = FakeSDK(["one", "two", "three"])
    client = make_client(tmp_path, sdk, daily_request_budget=2)
    client.generate([{"role": "user", "content": "q1"}])
    client.generate([{"role": "user", "content": "q2"}])

    with pytest.raises(BudgetExceededError):
        client.generate([{"role": "user", "content": "q3"}])
    assert len(sdk.requests) == 2  # the 3rd request was never sent

    # The counter lives on disk, so a new client sees the same usage.
    again = make_client(tmp_path, FakeSDK(["x"]), daily_request_budget=2)
    with pytest.raises(BudgetExceededError):
        again.generate([{"role": "user", "content": "q4"}])


def test_budget_resets_on_a_new_day(tmp_path):
    client = make_client(tmp_path, FakeSDK(["one"]), daily_request_budget=1)
    client.generate([{"role": "user", "content": "q1"}])

    tomorrow = make_client(
        tmp_path, FakeSDK(["two"]), daily_request_budget=1, today=lambda: date(2026, 11, 16)
    )
    assert tomorrow.requests_used_today() == 0
    assert tomorrow.generate([{"role": "user", "content": "q2"}]) == "two"


def test_minimum_gap_between_requests(tmp_path):
    clock = FakeClock()
    client = make_client(
        tmp_path, FakeSDK(["a", "b"]), clock=clock, min_seconds_between_requests=4.0
    )
    client.generate([{"role": "user", "content": "q1"}])
    client.generate([{"role": "user", "content": "q2"}])
    assert clock.sleeps == [4.0]


def test_missing_api_key_gives_clear_error(tmp_path):
    client = make_client(tmp_path, sdk=None, api_key=None)
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        client.generate(MESSAGES)


def test_blocked_response_without_text_raises(tmp_path):
    class NoTextSDK(FakeSDK):
        def generate_content(self, model, contents, config):
            return SimpleNamespace(text=None)

    client = make_client(tmp_path, NoTextSDK([]))
    with pytest.raises(LLMError, match="no text"):
        client.generate(MESSAGES)


def test_api_key_never_appears_in_repr_or_logs(tmp_path, caplog):
    secret = "AIza-very-secret-key"
    sdk = FakeSDK([FakeAPIError(429), "ok"])
    client = make_client(tmp_path, sdk, api_key=secret)
    with caplog.at_level(logging.DEBUG):
        client.generate(MESSAGES)
    assert secret not in repr(client)
    assert secret not in caplog.text
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert cache_files and all(secret not in f.read_text() for f in cache_files)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FakeAPIError(429), True),
        (FakeAPIError(503), True),
        (Exception("429 RESOURCE_EXHAUSTED quota"), True),
        (Exception("503 UNAVAILABLE"), True),
        (FakeAPIError(400), False),
        (Exception("prompt was 4290 tokens"), False),
    ],
)
def test_is_retryable_error(error, expected):
    assert is_retryable_error(error) is expected
