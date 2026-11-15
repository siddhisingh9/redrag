"""Export the judge training dataset: train / val / test JSONL files.

Where the labels come from:
- **hand**: your labels (`data/labels/hand_labels.jsonl`). Always used, and
  the only labels allowed in the test split, so the test measures agreement
  with a human, not with our own rules.
- **rules**: results where the rule judge is *confident* (a canary is there, the
  marker is missing, the model refused). These "weak" labels add volume to
  train/val. They are only used where the rules are trustworthy; "unsure"
  results without a hand label are left out.

No attack-id leakage (spec M4): the split is made by **seed**. A seed, all its
agent variants, and every run of them land in the same split. Otherwise the
model could see "dir-001" in training and a near-copy "dir-001.v2" in the
test, and the test score would be too optimistic. Seeds are spread per category,
so every split has every kind of attack.

Hand labels are shared out on purpose (`assign_splits_by_labels`): seeds that
have hand labels go roughly 45% train / 45% test / 10% val, balanced by a
weighted label count (hard cases count more, see `label_weight`). The first
version split seeds blindly, which put 127 of 150 hand labels in test. The
classifier then trained almost only on rule labels, learned to copy
the rules, and never saw a hard case (a paraphrased prompt, quoted staff notes).
It scored no better than chance on exactly those.

Each line of the output files:
    {"id", "seed_id", "category", "signal_type", "goal", "response",
     "label": 0 or 1 (1 = attack succeeded), "label_source": "hand" | "rules",
     "rules_verdict": what the rule judge said ("unsure" = a hard case)}
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from redrag.attacks.schema import CATEGORIES, AttackResult
from redrag.eval.judge_eval import attack_from_label
from redrag.eval.labels import (
    HandLabel,
    attack_from_result,
    content_key,
    result_key,
    seeds_by_id,
)
from redrag.judge.classifier import GOAL_TEXTS, goal_text, response_text
from redrag.judge.rules import judge_rules

SPLITS = ("train", "val", "test")
# Seeds per category are 11 in the shipped library -> 7 train, 1 val, 3 test.
DEFAULT_RATIOS = {"train": 0.65, "val": 0.10, "test": 0.25}


@dataclass
class Example:
    id: str
    seed_id: str
    category: str
    signal_type: str
    goal: str
    response: str
    label: int
    label_source: str
    rules_verdict: str = ""


# Share of the (weighted) hand labels each split should get. Dict order breaks
# ties: train first, because the rare hard successes help most in training.
HAND_LABEL_SHARES = {"train": 0.45, "test": 0.45, "val": 0.10}


def _hash(seed_id: str) -> str:
    return hashlib.md5(seed_id.encode("utf-8")).hexdigest()


def label_weight(example: Example) -> int:
    """How much a hand label counts when balancing the split.

    Hard cases (the rules were unsure) matter most, and hard *successes*
    (paraphrased prompts, quoted staff notes) are the rarest of all. Weighting
    them makes sure both training and test get a fair share.
    """
    if example.rules_verdict != "unsure":
        return 1
    return 6 if example.label else 3


def assign_splits_by_labels(
    seed_categories: dict[str, str],
    hand_counts: dict[str, int],
    shares: dict[str, float] = HAND_LABEL_SHARES,
) -> dict[str, str]:
    """Split seeds so the *hand labels* are shared between test, train and val.

    `hand_counts` is the (weighted) amount of hand labels per seed. Labelled
    seeds are handed out one at a time, heaviest first, to whichever split is
    furthest below its target share. This is done across *all* categories at
    once: a few seeds hold most of the hard cases, and balancing within each
    category separately sent each category's heaviest seed to test. Seeds
    without hand labels only carry rule labels, so they all go to train.
    Deterministic: ties are broken by a hash of the seed id.
    """
    labelled = [s for s in seed_categories if hand_counts.get(s, 0) > 0]
    labelled.sort(key=lambda s: (-hand_counts[s], _hash(s)))
    total = sum(hand_counts[s] for s in labelled)
    given = {split: 0 for split in shares}

    assignment: dict[str, str] = {}
    for seed_id in labelled:
        # The split with the biggest shortfall gets this seed (train first on ties).
        split = max(shares, key=lambda name: shares[name] * total - given[name])
        assignment[seed_id] = split
        given[split] += hand_counts[seed_id]
    for seed_id in seed_categories:
        assignment.setdefault(seed_id, "train")
    return assignment


def assign_splits(
    seed_categories: dict[str, str], ratios: dict[str, float] = DEFAULT_RATIOS
) -> dict[str, str]:
    """Map each seed id to "train" / "val" / "test", category by category.

    The order inside a category comes from a hash of the seed id, so it's
    random-looking but identical on every run and every computer.
    """
    by_category: dict[str, list[str]] = {}
    for seed_id, category in seed_categories.items():
        by_category.setdefault(category, []).append(seed_id)

    assignment: dict[str, str] = {}
    for seed_ids in by_category.values():
        ordered = sorted(seed_ids, key=lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
        n = len(ordered)
        n_test = max(1, round(n * ratios["test"])) if n >= 3 else 0
        n_val = max(1, round(n * ratios["val"])) if n >= 3 else 0
        for index, seed_id in enumerate(ordered):
            if index < n_test:
                assignment[seed_id] = "test"
            elif index < n_test + n_val:
                assignment[seed_id] = "val"
            else:
                assignment[seed_id] = "train"
    return assignment


def build_examples(
    results: list[AttackResult],
    labels: dict[str, HandLabel],
    system_prompt: str,
    use_rule_labels: bool = True,
) -> list[Example]:
    """One example per unique (goal, messages, answers). Hand labels beat rule labels."""
    seeds = seeds_by_id()
    examples: dict[tuple, Example] = {}

    # Hand labels first: they're self-contained, so they count even if runs/ was cleared.
    for label in labels.values():
        if label.label == "skip":
            continue
        signal = label.success_signal
        content = (signal.type, signal.value, tuple(label.turns), tuple(label.answers))
        examples[content] = Example(
            id=label.key,
            seed_id=label.seed_id,
            category=label.category,
            signal_type=signal.type,
            goal=goal_text(signal.type),
            response=response_text(label.answers),
            label=int(label.label == "success"),
            label_source="hand",
            # Re-judged with the *current* rules (they may have changed since labelling).
            rules_verdict=judge_rules(
                attack_from_label(label), label.transcript(), system_prompt
            ).verdict,
        )

    if use_rule_labels:
        for result in results:
            content = content_key(result)
            if content in examples or result.success_signal is None:
                continue
            attack = attack_from_result(result, seeds)
            verdict = judge_rules(attack, result.transcript, system_prompt)
            if verdict.verdict == "unsure":
                continue  # only trust the rules where they are confident
            examples[content] = Example(
                id=result_key(result),
                seed_id=result.root_id,
                category=result.category,
                signal_type=result.success_signal.type,
                goal=goal_text(result.success_signal.type),
                response=response_text([turn.answer for turn in result.transcript]),
                label=int(verdict.verdict == "success"),
                label_source="rules",
                rules_verdict=verdict.verdict,
            )
    return list(examples.values())


def split_examples(examples: list[Example]) -> dict[str, list[Example]]:
    """Group by seed into train/val/test. Rule-labelled examples never go to test.

    With hand labels, seeds are split so those labels are shared between test,
    train and val (`assign_splits_by_labels`). Without any, it falls back to a
    plain hash split (`assign_splits`).
    """
    seed_categories: dict[str, str] = {}
    weights: dict[str, int] = {}
    for example in examples:
        seed_categories.setdefault(example.seed_id, example.category)
        if example.label_source == "hand":
            weights[example.seed_id] = weights.get(example.seed_id, 0) + label_weight(example)
    if weights:
        assignment = assign_splits_by_labels(seed_categories, weights)
    else:
        assignment = assign_splits(seed_categories)

    splits: dict[str, list[Example]] = {name: [] for name in SPLITS}
    for example in sorted(examples, key=lambda e: e.id):
        split = assignment[example.seed_id]
        if split == "test" and example.label_source != "hand":
            continue
        splits[split].append(example)
    return splits


def export_dataset(
    results: list[AttackResult],
    labels: dict[str, HandLabel],
    system_prompt: str,
    out_dir: Path,
    use_rule_labels: bool = True,
) -> dict:
    """Write train/val/test.jsonl and dataset_info.json; return the info dict."""
    splits = split_examples(build_examples(results, labels, system_prompt, use_rule_labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "goal_texts": GOAL_TEXTS,
        "label_meaning": {"1": "attack succeeded", "0": "attack failed"},
        "splits": {},
    }
    for name, examples in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
        info["splits"][name] = {
            "examples": len(examples),
            "success": sum(e.label for e in examples),
            "fail": sum(1 - e.label for e in examples),
            "by_source": dict(Counter(e.label_source for e in examples)),
            "by_category": {c: sum(e.category == c for e in examples) for c in CATEGORIES},
            "seeds": sorted({e.seed_id for e in examples}),
        }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
