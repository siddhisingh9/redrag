"""Client for Google's Gemini API, used on the **free tier only**.

Gemini is the only remote model RedRAG uses, and only for one job: the backup
judge (when the local classifier is unsure).

The free tier is generous but strict, so this client protects it in three ways:

1. **Disk cache.** Each response is saved under a hash of the request
   (model + messages + options). Re-running a scan re-uses old answers and
   spends zero quota.
2. **Daily budget.** A small JSON file counts today's successful requests. When
   the count reaches `daily_request_budget`, we raise `BudgetExceededError`
   *before* calling the API.
3. **Backoff on 429.** If Google says "too many requests" (HTTP 429) or
   "overloaded" (HTTP 503), we wait and retry with exponential backoff:
   2s, 4s, 8s, ... up to `max_retries` times. We also leave a minimum gap
   between requests so we rarely hit the per-minute limit in the first place.

The API key is read from `GEMINI_API_KEY` and is never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from redrag.llm.base import (
    BudgetExceededError,
    LLMError,
    Message,
    RateLimitError,
    validate_messages,
)

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying: 429 = rate limited, 503 = model overloaded.
RETRYABLE_STATUS_CODES = (429, 503)
RETRYABLE_STATUS_NAMES = ("RESOURCE_EXHAUSTED", "UNAVAILABLE")


class GeminiClient:
    """Call a Gemini model with caching, a daily budget and retry-with-backoff."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        cache_dir: Path | str = ".cache/gemini",
        daily_request_budget: int = 200,
        min_seconds_between_requests: float = 4.0,
        max_retries: int = 5,
        backoff_base_seconds: float = 2.0,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
    ) -> None:
        """
        Args:
            model: Gemini model name, e.g. "gemini-2.5-flash-lite".
            api_key: Key from AI Studio. May be None if every request is a cache
                hit, or if a fake `sdk_client` is supplied (tests).
            cache_dir: Folder for cached responses and the budget counter.
            daily_request_budget: Max successful API calls per day.
            min_seconds_between_requests: Minimum gap between two API calls.
            max_retries: How many times to retry a 429/503 before giving up.
            backoff_base_seconds: First retry wait; doubles each retry.
            sdk_client: Optional pre-built client (tests pass a fake). If omitted,
                a real `google.genai.Client` is created on first use.
            sleep, monotonic, today: Injected so tests can control time
                without actually waiting.

        Note: Google resets free-tier daily quotas at midnight Pacific time; we
        reset our counter at local midnight. The budget default is set well
        below the real quota, so that difference doesn't matter in practice.
        """
        self.model = model
        self._api_key = api_key  # private: never logged or printed
        self.cache_dir = Path(cache_dir)
        self.daily_request_budget = daily_request_budget
        self.min_seconds_between_requests = min_seconds_between_requests
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._sdk_client = sdk_client
        self._sleep = sleep
        self._monotonic = monotonic
        self._today = today
        self._last_request_at: float | None = None

    def __repr__(self) -> str:
        # Custom repr so the API key can never end up in a log by accident.
        return f"GeminiClient(model={self.model!r}, budget={self.daily_request_budget}/day)"

    # ------------------------------------------------------------------ public

    def generate(self, messages: list[Message], **opts: Any) -> str:
        """Return Gemini's reply, from the cache if we've seen this exact request."""
        validate_messages(messages)

        cache_file = self.cache_dir / f"{self._cache_key(messages, opts)}.json"
        if cache_file.exists():
            logger.debug("Gemini cache hit: %s", cache_file.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        self._check_budget()
        text = self._call_with_retries(messages, opts)
        self._record_request()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"model": self.model, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )
        return text

    def requests_used_today(self) -> int:
        """How many API calls have been made today (cache hits don't count)."""
        state = self._read_budget_state()
        if state.get("date") != self._today().isoformat():
            return 0
        return int(state.get("used", 0))

    # --------------------------------------------------------------- the call

    def _call_with_retries(self, messages: list[Message], opts: dict[str, Any]) -> str:
        """Call the API, retrying 429/503 errors with exponential backoff."""
        system_instruction, contents = to_gemini_contents(messages)
        config = self._build_generation_config(system_instruction, opts)
        client = self._get_sdk_client()

        attempt = 0
        while True:
            self._wait_for_min_gap()
            try:
                response = client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as exc:  # the SDK raises its own error classes
                if not is_retryable_error(exc):
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                if attempt >= self.max_retries:
                    raise RateLimitError(
                        f"Gemini still rate-limited after {self.max_retries} retries"
                    ) from exc
                # Exponential backoff: base, 2*base, 4*base, ... plus a little
                # random "jitter" so parallel workers don't retry in lockstep.
                delay = self.backoff_base_seconds * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Gemini rate-limited/overloaded (attempt %d/%d); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
                attempt += 1
                continue

            text = getattr(response, "text", None)
            if text is None:
                # Happens e.g. when the response was blocked by Gemini's safety filter.
                raise LLMError("Gemini returned no text (the response may have been blocked)")
            return text

    def _build_generation_config(
        self, system_instruction: str | None, opts: dict[str, Any]
    ) -> dict[str, Any]:
        """Map our generic options to Gemini's config field names."""
        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if "temperature" in opts:
            config["temperature"] = opts["temperature"]
        if "max_tokens" in opts:
            config["max_output_tokens"] = opts["max_tokens"]
        if "seed" in opts:
            config["seed"] = opts["seed"]
        return config

    def _get_sdk_client(self) -> Any:
        """Create the real google-genai client lazily (only when actually needed).

        Importing lazily means the rest of RedRAG, and every test, works even if
        `google-genai` isn't installed or no key is set.
        """
        if self._sdk_client is None:
            if not self._api_key:
                raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
            try:
                from google import genai
            except ImportError as exc:
                raise LLMError(
                    "google-genai is not installed. Run: pip install -r requirements.txt"
                ) from exc
            self._sdk_client = genai.Client(api_key=self._api_key)
        return self._sdk_client

    def _wait_for_min_gap(self) -> None:
        """Sleep if the previous request was less than the minimum gap ago."""
        now = self._monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self.min_seconds_between_requests:
                self._sleep(self.min_seconds_between_requests - elapsed)
        self._last_request_at = self._monotonic()

    # ------------------------------------------------------------ the cache

    def _cache_key(self, messages: list[Message], opts: dict[str, Any]) -> str:
        """Hash everything that can change the answer into a short file name."""
        blob = json.dumps(
            {"model": self.model, "messages": messages, "opts": opts},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------- the budget

    @property
    def _budget_file(self) -> Path:
        return self.cache_dir / "budget.json"

    def _read_budget_state(self) -> dict[str, Any]:
        if not self._budget_file.exists():
            return {}
        try:
            return json.loads(self._budget_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _check_budget(self) -> None:
        used = self.requests_used_today()
        if used >= self.daily_request_budget:
            raise BudgetExceededError(
                f"Gemini daily budget used up ({used}/{self.daily_request_budget}). "
                "Raise GEMINI_DAILY_BUDGET or try again tomorrow."
            )

    def _record_request(self) -> None:
        """Add one to today's counter (a new day starts again from zero)."""
        used = self.requests_used_today() + 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_file.write_text(
            json.dumps({"date": self._today().isoformat(), "used": used}), encoding="utf-8"
        )


def to_gemini_contents(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert chat messages to Gemini's format.

    Gemini differs from the usual chat format in two ways:
    - the system prompt is passed separately as `system_instruction`, and
    - the assistant role is called "model".
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def is_retryable_error(exc: Exception) -> bool:
    """True if the error means "slow down" (429) or "busy, try later" (503).

    The google-genai SDK puts the HTTP status on `exc.code`. We check a few
    places rather than importing the SDK's error class, so this also works
    with the fake errors used in tests.
    """
    for attribute in ("code", "status_code"):
        if getattr(exc, attribute, None) in RETRYABLE_STATUS_CODES:
            return True
    message = str(exc)
    if any(name in message for name in RETRYABLE_STATUS_NAMES):
        return True
    # Match "429" / "503" as whole numbers only, so "4290 tokens" doesn't count.
    return re.search(r"\b(429|503)\b", message) is not None
