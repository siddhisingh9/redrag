"""Hand labels: the ground truth every judge layer is measured against.

A person reads a saved attack (goal, messages, answers) and decides whether
the attack really succeeded. Those labels:
- measure each judge layer (accuracy / F1 / Cohen's kappa), and
- are the most valuable training data for the fine-tuned classifier.

Labels live in `data/labels/hand_labels.jsonl`, one JSON object per line.
Each label stores the full conversation, not just a pointer into `runs/`,
because `runs/` is gitignored and may be deleted. The labels file should be
committed: it's hours of careful human work.

This module also loads saved results from all runs and rebuilds the Attack
each one came from, so old results can be re-judged and labelled.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from redrag.attacks.loader import load_attacks
from redrag.attacks.schema import Attack, AttackResult, PoisonDoc, SuccessSignal, Turn
from redrag.eval.report import RESULTS_FILE, load_results
from redrag.judge.rules import judge_rules

LabelValue = Literal["success", "fail", "skip"]


class HandLabel(BaseModel):
    """One human judgement of one attack result, with everything needed to reuse it."""

    key: str
    run_id: str
    attack_id: str
    seed_id: str
    defense_config: str
    category: str
    success_signal: SuccessSignal
    turns: list[str]
    answers: list[str]
    target_model: str
    label: LabelValue
    note: str = ""
    # What the rules said when this was labelled (hidden from the labeller, for analysis).
    rules_verdict: str = ""
    labelled_at: datetime
    # "blind": the person decided with no hint. "reviewed": they were shown a
    # suggested label first (and accepted or overrode it). Reports say which.
    method: Literal["blind", "reviewed"] = "blind"
    suggestion: LabelValue | None = None
    suggested_by: str = ""

    def transcript(self) -> list[Turn]:
        return [Turn(user=u, answer=a) for u, a in zip(self.turns, self.answers, strict=True)]


class Suggestion(BaseModel):
    """A suggested label shown to the person labelling (model-assisted labelling)."""

    key: str
    label: LabelValue
    reason: str
    suggested_by: str


def load_suggestions(path: Path) -> dict[str, Suggestion]:
    """Suggested labels by result key (empty if there is no suggestions file)."""
    suggestions: dict[str, Suggestion] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                suggestion = Suggestion.model_validate_json(line)
                suggestions[suggestion.key] = suggestion
    return suggestions


def result_key(result: AttackResult) -> str:
    """Unique id of one result: run + defense configuration + attack."""
    return f"{result.run_id}/{result.defense_config}/{result.attack_id}"


def content_key(result: AttackResult) -> tuple:
    """Two results with the same goal, messages and answers are the same labelling job."""
    signal = result.success_signal
    return (
        signal.type if signal else None,
        signal.value if signal else None,
        tuple(turn.user for turn in result.transcript),
        tuple(turn.answer for turn in result.transcript),
    )


# --------------------------------------------------------------------------
# Loading results
# --------------------------------------------------------------------------


def seeds_by_id() -> dict[str, Attack]:
    return {attack.id: attack for attack in load_attacks()}


def load_all_results(runs_dir: Path, include_mock: bool = False) -> list[AttackResult]:
    """Every judged result from every run, with the success signal filled in.

    Mock-target runs are skipped by default: their answers are scripted, so
    they teach a classifier nothing about real models. Errors are skipped too.
    """
    seeds = seeds_by_id()
    results: list[AttackResult] = []
    if not runs_dir.exists():
        return results
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / RESULTS_FILE).exists()):
        for result in load_results(run_dir):
            if result.final_verdict == "error" or not result.transcript:
                continue
            if not include_mock and result.model_names.get("target", "").startswith("mock"):
                continue
            if result.success_signal is None:  # runs saved before M4
                seed = seeds.get(result.root_id)
                if seed is None:
                    continue
                result.success_signal = seed.success_signal
            results.append(result)
    return results


def attack_from_result(result: AttackResult, seeds: dict[str, Attack] | None = None) -> Attack:
    """Rebuild the Attack a result came from (needed to re-judge it)."""
    seeds = seeds if seeds is not None else seeds_by_id()
    poison = None
    if result.poison_doc_id:
        text = result.poison_doc_text
        if text is None and result.root_id in seeds and seeds[result.root_id].poison_doc:
            text = seeds[result.root_id].poison_doc.text
        poison = PoisonDoc(doc_id=result.poison_doc_id, text=text or "(unknown)")
    signal = result.success_signal or seeds[result.root_id].success_signal
    return Attack(
        id=result.attack_id,
        category=result.category,
        owasp=result.owasp,
        language=result.language,
        turns=[turn.user for turn in result.transcript],
        poison_doc=poison,
        success_signal=signal,
        parent_id=result.parent_id,
        seed_id=result.seed_id,
        strategy=result.strategy,
    )


# --------------------------------------------------------------------------
# The label file
# --------------------------------------------------------------------------


def load_labels(path: Path) -> dict[str, HandLabel]:
    """All labels, by key. If a result was labelled twice, the latest label wins."""
    labels: dict[str, HandLabel] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            label = HandLabel.model_validate_json(line)
            labels[label.key] = label
    return labels


def append_label(path: Path, label: HandLabel) -> None:
    """Save one label immediately, so quitting halfway loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(label.model_dump_json() + "\n")


def make_label(
    result: AttackResult,
    label: LabelValue,
    system_prompt: str,
    note: str = "",
    suggestion: Suggestion | None = None,
) -> HandLabel:
    """Build a label. Pass the `suggestion` that was shown, if any (method = "reviewed")."""
    rules = judge_rules(attack_from_result(result), result.transcript, system_prompt)
    assert result.success_signal is not None
    return HandLabel(
        method="reviewed" if suggestion else "blind",
        suggestion=suggestion.label if suggestion else None,
        suggested_by=suggestion.suggested_by if suggestion else "",
        key=result_key(result),
        run_id=result.run_id,
        attack_id=result.attack_id,
        seed_id=result.root_id,
        defense_config=result.defense_config,
        category=result.category,
        success_signal=result.success_signal,
        turns=[turn.user for turn in result.transcript],
        answers=[turn.answer for turn in result.transcript],
        target_model=result.model_names.get("target", "unknown"),
        label=label,
        note=note,
        rules_verdict=rules.verdict,
        labelled_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------
# Choosing what to label
# --------------------------------------------------------------------------


def _stable_rank(result: AttackResult) -> str:
    """A random-looking but fixed position for each result.

    A hash of the result's key doesn't depend on which *other* results are still
    unlabelled, so the queue order stays the same between labelling sessions.
    """
    return hashlib.md5(result_key(result).encode("utf-8")).hexdigest()


def _interleave(groups: dict[tuple[str, str], list[AttackResult]]) -> list[AttackResult]:
    """Take one from each group in turn, so every kind of attack/outcome is covered early."""
    ordered = {key: sorted(members, key=_stable_rank) for key, members in sorted(groups.items())}
    queue: list[AttackResult] = []
    index = 0
    while any(index < len(members) for members in ordered.values()):
        for members in ordered.values():
            if index < len(members):
                queue.append(members[index])
        index += 1
    return queue


def labelling_queue(
    results: list[AttackResult],
    labels: dict[str, HandLabel],
    system_prompt: str,
) -> list[AttackResult]:
    """Order unlabelled results so the most useful ones come first.

    1. Duplicates are dropped (same goal, messages and answers). So are results
       whose content was already labelled under another key.
    2. Everything the rules call "unsure" comes first: the hard cases the
       classifier must learn, and where the judges disagree most.
    3. Then results from **test-split** seeds: only those grade the judges in
       `redrag judge-eval`, so labelling them first makes every label count.
    4. Then everything else.
    Within steps 3 and 4, results are interleaved across (category, rules
    verdict) groups, so the labels cover every kind of attack and outcome instead
    of 150 near-identical direct injections. The order is stable between sessions.
    """
    from redrag.eval.dataset import assign_splits  # here, to avoid a circular import

    seeds = seeds_by_id()
    splits = assign_splits({seed_id: seed.category for seed_id, seed in seeds.items()})
    labelled_content = set()
    for label in labels.values():
        labelled_content.add(
            (
                label.success_signal.type,
                label.success_signal.value,
                tuple(label.turns),
                tuple(label.answers),
            )
        )

    unique: dict[tuple, AttackResult] = {}
    for result in results:
        content = content_key(result)
        if result_key(result) in labels or content in labelled_content or content in unique:
            continue
        unique[content] = result

    unsure: list[AttackResult] = []
    test_groups: dict[tuple[str, str], list[AttackResult]] = {}
    other_groups: dict[tuple[str, str], list[AttackResult]] = {}
    for result in unique.values():
        verdict = judge_rules(attack_from_result(result, seeds), result.transcript, system_prompt)
        if verdict.verdict == "unsure":
            unsure.append(result)
            continue
        groups = test_groups if splits.get(result.root_id) == "test" else other_groups
        groups.setdefault((result.category, verdict.verdict), []).append(result)

    return sorted(unsure, key=_stable_rank) + _interleave(test_groups) + _interleave(other_groups)
