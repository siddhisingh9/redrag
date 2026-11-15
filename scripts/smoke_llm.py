"""Smoke test: can RedRAG reach the real models on this machine?

Run:  python scripts/smoke_llm.py

It tries Ollama and Gemini once each with a tiny prompt. Anything that isn't
available (Ollama not running, model not pulled, no API key, SDK missing) is
reported as SKIP rather than as a crash, so this is safe to run anywhere.
Exit code is 1 only if a service *is* available but the call failed.

The Gemini call is cached, so running this script again costs no quota.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let `python scripts/smoke_llm.py` import the package without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redrag.config import load_config  # noqa: E402
from redrag.llm import (  # noqa: E402
    LLMError,
    OllamaClient,
    build_gemini_client,
)

PROMPT = [
    {"role": "system", "content": "You answer with exactly one word."},
    {"role": "user", "content": "Reply with the single word: pong"},
]


def check_ollama(config) -> str:
    """Return "OK", "SKIP" or "FAIL" for the victim and attacker models."""
    models = {config.ollama.victim_model, config.ollama.attacker_model}
    status = "OK"
    for model in sorted(models):
        client = OllamaClient(model=model, host=config.ollama.host, timeout_seconds=180)
        if not client.is_available():
            print(f"[SKIP] Ollama is not running at {config.ollama.host}. Start the Ollama app.")
            return "SKIP"
        if not client.has_model():
            print(f"[SKIP] Ollama model {model!r} is not pulled. Run: ollama pull {model}")
            status = "SKIP"
            continue
        try:
            reply = client.generate(PROMPT, max_tokens=10, temperature=0)
        except LLMError as exc:
            print(f"[FAIL] Ollama {model}: {exc}")
            return "FAIL"
        print(f"[OK]   Ollama {model} replied: {reply.strip()[:60]!r}")
    return status


def check_gemini(config) -> str:
    """Return "OK", "SKIP" or "FAIL" for the configured Gemini model."""
    if config.gemini.api_key is None:
        print("[SKIP] GEMINI_API_KEY is not set (add it to .env to enable Gemini).")
        return "SKIP"
    try:
        import google.genai  # noqa: F401
    except ImportError:
        print("[SKIP] google-genai is not installed. Run: pip install -r requirements.txt")
        return "SKIP"

    client = build_gemini_client(config)
    try:
        reply = client.generate(PROMPT, max_tokens=10, temperature=0)
    except LLMError as exc:
        print(f"[FAIL] Gemini {config.gemini.model}: {exc}")
        print("       Check the model name in AI Studio and set GEMINI_MODEL in .env.")
        return "FAIL"
    used = client.requests_used_today()
    print(
        f"[OK]   Gemini {config.gemini.model} replied: {reply.strip()[:60]!r} "
        f"({used}/{config.gemini.daily_request_budget} requests used today)"
    )
    return "OK"


def main() -> int:
    config = load_config()
    results = [check_ollama(config), check_gemini(config)]
    return 1 if "FAIL" in results else 0


if __name__ == "__main__":
    sys.exit(main())
