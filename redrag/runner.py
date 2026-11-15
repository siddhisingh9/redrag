"""Orchestrates a scan. Built in M2.

For each attack: reset the target, ingest any poison document, send the turns
with the chosen defenses, judge the answer and append an `AttackResult` to
`runs/<timestamp>/results.jsonl`.
"""
