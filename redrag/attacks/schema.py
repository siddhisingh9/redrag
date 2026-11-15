"""Pydantic models for attacks and their results. Built in M2.

- `Attack`: one test case (id, category, OWASP tag, language, turns, optional
  poison_doc, success_signal, notes). Stored one-per-line in `seeds/*.jsonl`.
- `AttackResult`: what happened when we ran it (defenses on, transcript,
  retrieved doc ids, each judge layer's verdict, final verdict, latency,
  model names, timestamp).
"""
