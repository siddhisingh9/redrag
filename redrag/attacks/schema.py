"""Pydantic models for attacks and their results.

- `Attack`: one test case, stored one-per-line in `seeds/*.jsonl`.
- `AttackResult`: what happened when we ran one attack against the target
  with one set of defenses. Results are saved to `runs/<id>/results.jsonl`.

Validating with pydantic means a typo in a seed file (a wrong category, a
missing poison document) is caught when the file is loaded, with a clear message,
not halfway through a scan.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "direct_injection",
    "indirect_injection",
    "system_prompt_extraction",
    "data_exfiltration",
    "multi_turn_escalation",
]
CATEGORIES: tuple[str, ...] = (
    "direct_injection",
    "indirect_injection",
    "system_prompt_extraction",
    "data_exfiltration",
    "multi_turn_escalation",
)

Verdict = Literal["success", "fail", "unsure"]
FinalVerdict = Literal["success", "fail", "unsure", "error"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PoisonDoc(_Strict):
    """A document the attacker slips into the knowledge base via `/ingest`."""

    doc_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    text: str = Field(min_length=1)


class SuccessSignal(_Strict):
    """How to tell that the attack worked.

    - marker: the answer contains the marker word in `value` (e.g. PINEAPPLE-7).
    - canary: the answer contains a canary. `value` names the one the attack is
      after (informational: leaking any canary counts as a leak).
    - system_prompt: the answer reveals the system prompt (its canary, or a
      long stretch of its wording).
    """

    type: Literal["marker", "canary", "system_prompt"]
    value: str | None = None

    @model_validator(mode="after")
    def _marker_needs_value(self) -> SuccessSignal:
        if self.type == "marker" and not self.value:
            raise ValueError("a 'marker' success signal needs a value, e.g. PINEAPPLE-7")
        return self


class Attack(_Strict):
    """One attack test case."""

    # Letters, digits, dots, dashes, underscores; e.g. "ind-001" or "ind-001.v2" (M3).
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    category: Category
    # OWASP Top 10 for LLM Applications (2025): LLM01 prompt injection,
    # LLM02 sensitive information disclosure, LLM07 system prompt leakage.
    owasp: str = "LLM01"
    language: str = "en"
    # The user messages, sent in order within one chat session.
    turns: list[str] = Field(min_length=1, max_length=5)
    poison_doc: PoisonDoc | None = None
    success_signal: SuccessSignal
    notes: str = ""
    # Set on variants created by the attacker agent (M3), for lineage.
    parent_id: str | None = None

    @model_validator(mode="after")
    def _check_category_rules(self) -> Attack:
        if any(not turn.strip() for turn in self.turns):
            raise ValueError("turns must not be empty strings")
        if self.category == "indirect_injection" and self.poison_doc is None:
            raise ValueError("indirect_injection attacks need a poison_doc")
        if self.category == "multi_turn_escalation" and not 2 <= len(self.turns) <= 5:
            raise ValueError("multi_turn_escalation attacks need 2 to 5 turns")
        return self


class Turn(_Strict):
    """One message sent to the target and what came back."""

    user: str
    answer: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    blocked_by: str | None = None
    latency_ms: float = 0.0


class JudgeVerdict(_Strict):
    """One judge layer's opinion. `score` is a probability for the classifier (M4)."""

    layer: str
    verdict: Verdict
    reason: str = ""
    score: float | None = None


class AttackResult(_Strict):
    """Everything recorded about one attack run against one defense configuration."""

    run_id: str
    attack_id: str
    parent_id: str | None = None
    category: Category
    owasp: str
    language: str
    # Label used for grouping in reports: "none", "all", or "a,b".
    defense_config: str
    # The defenses the target says were active (from its /chat response).
    defenses: list[str]
    poison_doc_id: str | None = None
    transcript: list[Turn]
    retrieved_doc_ids: list[str]
    judge_verdicts: list[JudgeVerdict]
    final_verdict: FinalVerdict
    decided_by: str
    latency_ms: float
    model_names: dict[str, str]
    timestamp: datetime
    error: str | None = None
