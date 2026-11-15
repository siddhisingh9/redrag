"""Loads and validates seed attacks from `seeds/*.jsonl`.

JSONL = one JSON object per line. It's easy to add, remove or diff single
attacks, and a broken line only needs its line number to be found.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from redrag.attacks.schema import CATEGORIES, Attack

SEEDS_DIR = Path(__file__).resolve().parent / "seeds"


class AttackFileError(ValueError):
    """A seed file has a line that isn't a valid attack."""


def load_attack_file(path: Path) -> list[Attack]:
    """Load one JSONL file. Blank lines and lines starting with `//` are skipped."""
    attacks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            attacks.append(Attack.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise AttackFileError(f"{path.name}, line {line_number}: {exc}") from exc
    return attacks


def load_attacks(
    seeds_dir: Path = SEEDS_DIR,
    categories: Iterable[str] | None = None,
) -> list[Attack]:
    """Load every seed file, optionally keeping only some categories.

    Raises AttackFileError on an invalid line, a duplicate id, or an unknown
    category name in `categories`.
    """
    wanted = set(categories) if categories is not None else set(CATEGORIES)
    unknown = wanted - set(CATEGORIES)
    if unknown:
        raise AttackFileError(f"Unknown categories: {sorted(unknown)}. Choose from {CATEGORIES}.")

    attacks: list[Attack] = []
    seen_ids: dict[str, str] = {}
    for path in sorted(seeds_dir.glob("*.jsonl")):
        for attack in load_attack_file(path):
            if attack.id in seen_ids:
                raise AttackFileError(
                    f"Duplicate attack id {attack.id!r} in {path.name} "
                    f"(already in {seen_ids[attack.id]})"
                )
            seen_ids[attack.id] = path.name
            if attack.category in wanted:
                attacks.append(attack)
    return attacks


def select_attacks(attacks: list[Attack], max_attacks: int | None) -> list[Attack]:
    """Pick up to `max_attacks`, taking turns between categories.

    Round-robin selection means a small scan (e.g. --max-attacks 10) still
    covers every category instead of only the first file.
    """
    if max_attacks is None or max_attacks >= len(attacks):
        return list(attacks)

    by_category: dict[str, list[Attack]] = {}
    for attack in attacks:
        by_category.setdefault(attack.category, []).append(attack)

    selected: list[Attack] = []
    index = 0
    while len(selected) < max_attacks:
        for category_attacks in by_category.values():
            if index < len(category_attacks) and len(selected) < max_attacks:
                selected.append(category_attacks[index])
        index += 1
    return selected
