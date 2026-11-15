"""Loads and validates seed attacks from `seeds/*.jsonl`. Built in M2.

Every line is validated against the `Attack` schema, so a broken seed fails
with a clear message (file + line number) instead of crashing mid-scan.
"""
