"""Tests for the ASR math."""

from __future__ import annotations

from datetime import datetime, timezone

from redrag.attacks.schema import AttackResult
from redrag.eval.metrics import (
    ASRStat,
    asr_table,
    best_per_seed,
    compute_asr,
    defense_configs,
    group_asr,
    original_attempts,
)


def result(verdict: str, category: str = "direct_injection", config: str = "none"):
    return AttackResult(
        run_id="r",
        attack_id="a-001",
        category=category,
        owasp="LLM01",
        language="en",
        defense_config=config,
        defenses=[],
        transcript=[],
        retrieved_doc_ids=[],
        judge_verdicts=[],
        final_verdict=verdict,
        decided_by="rules",
        latency_ms=0,
        model_names={},
        timestamp=datetime.now(timezone.utc),
    )


def test_asr_counts_success_only_and_skips_errors():
    stat = compute_asr(
        [result("success"), result("fail"), result("unsure"), result("success"), result("error")]
    )
    assert (stat.successes, stat.fails, stat.unsure, stat.errors) == (2, 1, 1, 1)
    assert stat.judged == 4
    assert stat.asr == 0.5
    assert stat.format() == "2/4 (50%), 1 unsure"


def test_empty_and_error_only_have_no_asr():
    assert ASRStat().asr is None
    assert ASRStat().format() == "n/a"
    assert compute_asr([result("error")]).asr is None


def test_group_asr_by_category():
    groups = group_asr(
        [result("success", "direct_injection"), result("fail", "data_exfiltration")],
        key=lambda r: r.category,
    )
    assert groups["direct_injection"].asr == 1.0
    assert groups["data_exfiltration"].asr == 0.0


def test_asr_table_before_vs_after():
    results = [
        result("success", "direct_injection", "none"),
        result("success", "data_exfiltration", "none"),
        result("fail", "direct_injection", "all"),
        result("success", "data_exfiltration", "all"),
    ]
    table = asr_table(results)
    assert list(table) == ["direct_injection", "data_exfiltration", "overall"]
    assert table["direct_injection"]["none"].asr == 1.0
    assert table["direct_injection"]["all"].asr == 0.0
    assert table["overall"]["none"].asr == 1.0
    assert table["overall"]["all"].asr == 0.5


def test_defense_configs_keep_run_order():
    results = [
        result("fail", config="all"),
        result("fail", config="none"),
        result("fail", config="all"),
    ]
    assert defense_configs(results) == ["all", "none"]


def test_best_per_seed_prefers_success_then_unsure_then_fail():
    rows = [result("fail"), result("unsure"), result("success"), result("fail")]
    for index, row in enumerate(rows):
        row.attempt = index
        row.seed_id = "dir-001"
    assert [r.final_verdict for r in best_per_seed(rows)] == ["success"]
    assert [r.final_verdict for r in best_per_seed(rows[:2] + rows[3:])] == ["unsure"]
    assert [r.attempt for r in original_attempts(rows)] == [0]


def test_best_per_seed_keeps_configs_apart():
    rows = [result("success", config="none"), result("fail", config="all")]
    for row in rows:
        row.seed_id = "dir-001"
    assert sorted(r.defense_config for r in best_per_seed(rows)) == ["all", "none"]
