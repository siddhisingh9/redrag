"""Combines the judge layers: rules -> classifier (M4) -> Gemini (M4).

In M2 only the rule judge exists, so its verdict is final and "unsure" stays
"unsure". The report counts those separately from successes and failures.
M4 adds the fine-tuned classifier and the Gemini backup judge to settle them.

The result keeps both the final verdict and which layer decided it, so the
report can show how often each layer was needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from redrag.attacks.schema import Attack, FinalVerdict, JudgeVerdict, Turn
from redrag.judge.rules import judge_rules


@dataclass
class JudgeDecision:
    verdicts: list[JudgeVerdict]
    final_verdict: FinalVerdict
    decided_by: str


class JudgePipeline:
    """Runs the judge layers in order and stops at the first confident answer."""

    def __init__(self, system_prompt: str, overlap_threshold: float = 0.5) -> None:
        self.system_prompt = system_prompt
        self.overlap_threshold = overlap_threshold

    @property
    def layers(self) -> list[str]:
        """Names of the layers in use, for the report."""
        return ["rules"]

    def judge(self, attack: Attack, transcript: list[Turn]) -> JudgeDecision:
        rules_verdict = judge_rules(attack, transcript, self.system_prompt, self.overlap_threshold)
        return JudgeDecision(
            verdicts=[rules_verdict],
            final_verdict=rules_verdict.verdict,
            decided_by="rules",
        )
