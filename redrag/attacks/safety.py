"""Keeps generated attacks benign (spec rule 0.7).

RedRAG attacks only try to make the chatbot say a marker word, leak a fake
canary, or reveal its system prompt. The attacker agent (M3) uses an LLM to
rewrite attacks, and an LLM can drift. Every generated variant goes through
`check_variant` before it is sent; anything that strays toward harmful topics
is thrown away and replaced by a safe template variant.

A keyword list can't catch every harmful idea, but here it doesn't need to.
The attacker is told to stay benign, and this is the backstop for obvious drift.
"""

from __future__ import annotations

import re

# Topics a RedRAG attack never has a reason to mention.
HARMFUL_TERMS = [
    r"weapons?",
    r"bombs?",
    r"explosives?",
    r"guns?",
    r"poison(ing)? (someone|a person|people)",
    r"malware",
    r"ransomware",
    r"viruse?s?",
    r"keylogger",
    r"hack(ing)? into",
    r"steal(ing)? (passwords|credit cards?|money)",
    r"credit card numbers?",
    r"suicide",
    r"self[- ]harm",
    r"kill(ing)?",
    r"murder",
    r"drugs?",
    r"meth",
    r"racist",
    r"slurs?",
    r"porn",
    r"sexual",
    r"nude",
    r"terroris[mt]",
]
_HARMFUL_RE = re.compile(r"\b(" + "|".join(HARMFUL_TERMS) + r")\b", re.IGNORECASE)

MAX_TURN_CHARS = 1000


def find_harmful_terms(text: str) -> list[str]:
    """Return the harmful terms found in `text` (empty list = looks benign)."""
    return sorted({match.group(0).lower() for match in _HARMFUL_RE.finditer(text)})


def check_variant(turns: list[str], poison_text: str | None, marker: str | None) -> str | None:
    """Return a reason the variant is unusable, or None if it's fine.

    Checks: 1-5 non-empty turns of reasonable length, no harmful terms, and,
    for marker attacks, that the marker word is still somewhere in the attack
    (otherwise a "success" could never be detected).
    """
    if not 1 <= len(turns) <= 5:
        return f"needs 1-5 turns, got {len(turns)}"
    if any(not turn.strip() for turn in turns):
        return "empty turn"
    if any(len(turn) > MAX_TURN_CHARS for turn in turns):
        return f"turn longer than {MAX_TURN_CHARS} characters"

    all_text = " ".join(turns) + " " + (poison_text or "")
    harmful = find_harmful_terms(all_text)
    if harmful:
        return f"mentions off-limits topics: {', '.join(harmful)}"
    if marker and marker.lower() not in all_text.lower():
        return f"marker {marker} was dropped"
    return None
