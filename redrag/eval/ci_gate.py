"""The CI red-team gate: fail a pull request if the defenses got weaker.

On every pull request, GitHub Actions starts the target app with the MockClient,
runs a fixed scan and calls `redrag ci-check`. This module decides pass or fail
by comparing the scan with `config/ci_baseline.json`.

Why the scan has to be deterministic:
- The MockClient always gives the same answer to the same input, and the
  hashing embedder always retrieves the same chunks.
- The judge is rules-only. The trained classifier isn't in the repo (and would
  differ between machines), and Gemini needs a key and quota. The rules are
  plain code.
So the same code always gives the same ASR, and any change in the number comes
from a change in the code, not from noise. That's why the threshold can sit just
above today's value: one extra attack getting through is enough to fail.

Checks:
1. **ASR with all defenses on must not rise above `max_asr_all_defenses`.** This
   is the actual gate: a PR that weakens a defense lets more attacks through.
2. **ASR with no defenses must stay above `min_asr_no_defenses`.** This is a
   sanity check on the scanner itself. If the attacks stop working even against
   an undefended app, the attacks or the judge broke (e.g. a judge that never
   says "success" would pass check 1 forever).
3. **No errors**, and **"all" means the defenses the baseline was recorded with.**
   Otherwise the numbers aren't comparable, e.g. if judge_guard was switched on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from redrag.attacks.schema import AttackResult
from redrag.eval.metrics import ASRStat, compute_asr

DEFAULT_BASELINE = "config/ci_baseline.json"


@dataclass
class Baseline:
    max_asr_all_defenses: float
    min_asr_no_defenses: float
    expected_defenses: list[str]
    categories: list[str] | None = None  # None = every category
    max_attacks: int | None = None  # None = every seed attack
    recorded: dict = field(default_factory=dict)


def load_baseline(path: Path) -> Baseline:
    data = json.loads(path.read_text(encoding="utf-8"))
    scan = data.get("scan", {})
    return Baseline(
        max_asr_all_defenses=float(data["max_asr_all_defenses"]),
        min_asr_no_defenses=float(data["min_asr_no_defenses"]),
        expected_defenses=sorted(data["expected_defenses"]),
        categories=scan.get("categories"),
        max_attacks=scan.get("max_attacks"),
        recorded=data.get("recorded", {}),
    )


@dataclass
class GateResult:
    passed: bool
    asr_none: ASRStat
    asr_all: ASRStat
    problems: list[str]  # why it failed (empty if it passed)
    notes: list[str]  # extra hints that don't fail the gate


def check(results: list[AttackResult], baseline: Baseline) -> GateResult:
    """Compare one scan (configurations "none" and "all") with the baseline."""
    none = [r for r in results if r.defense_config == "none"]
    all_on = [r for r in results if r.defense_config == "all"]
    asr_none, asr_all = compute_asr(none), compute_asr(all_on)
    problems: list[str] = []
    notes: list[str] = []

    if not none or not all_on:
        problems.append('The scan must include both the "none" and "all" configurations.')
        return GateResult(False, asr_none, asr_all, problems, notes)

    errors = asr_none.errors + asr_all.errors
    if errors:
        problems.append(f"{errors} attack(s) ended in an error, so the scan is incomplete.")

    active = sorted(next((r.defenses for r in all_on if r.defenses), []))
    if active != baseline.expected_defenses:
        problems.append(
            f'"all" turned on {active}, but the baseline was recorded with '
            f"{baseline.expected_defenses}, so the numbers aren't comparable. "
            "(Start the target with `--no-guard` to leave judge_guard out.)"
        )

    if asr_all.asr is not None and asr_all.asr > baseline.max_asr_all_defenses:
        problems.append(
            f"Regression: ASR with all defenses is {asr_all.asr:.1%} "
            f"({asr_all.successes}/{asr_all.judged}), above the allowed "
            f"{baseline.max_asr_all_defenses:.1%}. A change let more attacks through."
        )
    if asr_none.asr is not None and asr_none.asr < baseline.min_asr_no_defenses:
        problems.append(
            f"ASR with no defenses is only {asr_none.asr:.1%}, below "
            f"{baseline.min_asr_no_defenses:.1%}. The attacks or the judge look broken "
            "(they should succeed against an undefended app)."
        )

    recorded = baseline.recorded.get("asr_all_defenses")
    improved = recorded is not None and asr_all.asr is not None and asr_all.asr < recorded
    if not problems and improved:
        notes.append(
            f"Improvement: ASR with all defenses fell from {recorded:.1%} to "
            f"{asr_all.asr:.1%}. Consider lowering max_asr_all_defenses in the "
            "baseline so the gain is locked in."
        )
    return GateResult(not problems, asr_none, asr_all, problems, notes)
