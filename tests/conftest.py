"""Shared pytest setup.

Tests must run offline with no API keys (spec rule 0.9). This autouse fixture
removes every RedRAG-related environment variable before each test, so a
developer's real shell settings can never change test results or spend quota.
"""

from __future__ import annotations

import pytest

from redrag.config import ENV_OVERRIDES


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [*ENV_OVERRIDES, "REDRAG_CONFIG"]:
        monkeypatch.delenv(name, raising=False)
