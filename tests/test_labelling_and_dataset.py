"""Tests for M4's data workflow: hand labels -> dataset export -> judge evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from redrag.attacks.loader import load_attacks
from redrag.attacks.schema import JudgeVerdict
from redrag.cli import app as cli_app
from redrag.config import load_config
from redrag.eval.dataset import assign_splits, build_examples, export_dataset, split_examples
from redrag.eval.judge_eval import evaluate_judges, render_agreement_markdown
from redrag.eval.labels import (
    HandLabel,
    append_label,
    attack_from_result,
    labelling_queue,
    load_all_results,
    load_labels,
    make_label,
    result_key,
)
from redrag.eval.report import load_results, write_markdown_report
from redrag.judge.pipeline import JudgePipeline
from redrag.judge.rules import judge_rules
from redrag.llm import MockClient
from redrag.runner import TargetClient, rejudge, run_scan
from redrag.target.app import create_app
from redrag.target.rag import SYSTEM_PROMPT

runner = CliRunner()


@pytest.fixture
def runs_dir(pipeline, tmp_path):
    """A runs folder holding one small mock scan (10 attacks x 2 configurations)."""
    attacks = load_attacks()[::5] + load_attacks(categories=["data_exfiltration"])[:3]
    app = create_app(load_config(env={}), MockClient(model="mock-victim"), pipeline)
    run_scan(
        attacks,
        TargetClient(TestClient(app)),
        JudgePipeline(SYSTEM_PROMPT),
        ["none", "all"],
        runs_dir=tmp_path / "runs",
        target_url="test",
        run_id="r1",
    )
    return tmp_path / "runs"


def results(runs_dir):
    return load_all_results(runs_dir, include_mock=True)


# ---------------------------------------------------------------- loading


def test_mock_runs_are_skipped_unless_asked(runs_dir):
    assert load_all_results(runs_dir) == []
    assert len(results(runs_dir)) == 28


def test_old_results_get_their_success_signal_from_the_seeds(runs_dir):
    path = runs_dir / "r1" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("success_signal")  # like a run saved before M4
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = results(runs_dir)
    assert all(r.success_signal is not None for r in loaded)


# ------------------------------------------------------------------ labels


def test_labels_are_self_contained_and_latest_wins(runs_dir, tmp_path):
    result = results(runs_dir)[0]
    path = tmp_path / "labels.jsonl"
    append_label(path, make_label(result, "fail", SYSTEM_PROMPT))
    append_label(path, make_label(result, "success", SYSTEM_PROMPT, note="changed my mind"))

    labels = load_labels(path)
    assert list(labels) == [result_key(result)]
    label = labels[result_key(result)]
    assert label.label == "success" and label.note == "changed my mind"
    assert label.answers == [turn.answer for turn in result.transcript]
    assert label.rules_verdict in ("success", "fail", "unsure")
    assert label.transcript()[0].user == result.transcript[0].user


def test_queue_puts_unsure_first_and_skips_duplicates_and_done(runs_dir, tmp_path):
    loaded = results(runs_dir)
    queue = labelling_queue(loaded, {}, SYSTEM_PROMPT)
    contents = [
        (tuple(t.user for t in r.transcript), tuple(t.answer for t in r.transcript)) for r in queue
    ]
    assert len(contents) == len(set(contents))  # no duplicate labelling jobs
    assert queue == labelling_queue(loaded, {}, SYSTEM_PROMPT)  # same order every time

    # Everything the rules call "unsure" comes before everything they are sure about.
    verdicts = [
        judge_rules(attack_from_result(r), r.transcript, SYSTEM_PROMPT).verdict for r in queue
    ]
    assert "unsure" in verdicts and verdicts[0] == "unsure"
    last_unsure = max(i for i, v in enumerate(verdicts) if v == "unsure")
    assert all(v == "unsure" for v in verdicts[: last_unsure + 1])

    labelled = make_label(queue[0], "fail", SYSTEM_PROMPT)
    later = labelling_queue(loaded, {labelled.key: labelled}, SYSTEM_PROMPT)
    assert result_key(queue[0]) not in {result_key(r) for r in later}
    assert len(later) == len(queue) - 1


# ----------------------------------------------------------------- dataset


def test_assign_splits_is_grouped_deterministic_and_covers_categories():
    seeds = {a.id: a.category for a in load_attacks()}
    first, second = assign_splits(seeds), assign_splits(seeds)
    assert first == second
    for category in set(seeds.values()):
        in_category = [first[s] for s, c in seeds.items() if c == category]
        assert in_category.count("test") == 3 and in_category.count("val") == 1


def fake_label(key, seed_id, label, answer, category="data_exfiltration"):
    signal = {"type": "canary", "value": "CANARY-DOC7X3"}
    return HandLabel(
        key=key,
        run_id="r",
        attack_id=seed_id,
        seed_id=seed_id,
        defense_config="none",
        category=category,
        success_signal=signal,
        turns=["q"],
        answers=[answer],
        target_model="m",
        label=label,
        labelled_at=datetime.now(timezone.utc),
    )


def test_examples_prefer_hand_labels_and_test_holds_only_hand_labels(runs_dir):
    loaded = results(runs_dir)
    hand = {
        result_key(r): make_label(r, "success", SYSTEM_PROMPT) for r in loaded[:4]
    }  # deliberately disagree with the rules sometimes
    examples = build_examples(loaded, hand, SYSTEM_PROMPT)
    sources = {e.id: e.label_source for e in examples}
    for key in hand:
        assert sources.get(key) == "hand" or key not in sources  # dupes collapse

    splits = split_examples(examples)
    assert all(e.label_source == "hand" for e in splits["test"])
    seeds_per_split = {name: {e.seed_id for e in items} for name, items in splits.items()}
    assert not seeds_per_split["train"] & seeds_per_split["test"]  # no attack-id leakage
    assert not seeds_per_split["train"] & seeds_per_split["val"]
    assert not seeds_per_split["val"] & seeds_per_split["test"]


def test_rule_labels_can_be_turned_off_and_skips_are_ignored(runs_dir):
    loaded = results(runs_dir)
    skip = make_label(loaded[0], "skip", SYSTEM_PROMPT)
    assert build_examples(loaded, {skip.key: skip}, SYSTEM_PROMPT, use_rule_labels=False) == []


def test_export_writes_files_and_info(runs_dir, tmp_path):
    info = export_dataset(results(runs_dir), {}, SYSTEM_PROMPT, tmp_path / "ds")
    for name in ("train", "val", "test"):
        assert (tmp_path / "ds" / f"{name}.jsonl").exists()
    assert info["splits"]["test"]["examples"] == 0  # no hand labels yet
    first_line = (tmp_path / "ds" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    example = json.loads(first_line)
    assert set(example) == {
        "id", "seed_id", "category", "signal_type", "goal", "response", "label", "label_source",
        "rules_verdict",
    }  # fmt: skip
    assert example["label"] in (0, 1)


# ------------------------------------------------------------- evaluation


class FixedClassifier:
    name = "fixed"

    def score(self, signal_type, answers):
        return 0.9 if "staff" in " ".join(answers).lower() else 0.1


class FixedGemini:
    name = "fixed-gemini"

    def judge(self, signal, transcript):
        return JudgeVerdict(layer="gemini", verdict="fail", reason="")


def test_evaluate_judges_scores_every_layer():
    labels = [
        fake_label("a", "exf-001", "success", "The staff code is in the safe notes."),
        fake_label("b", "exf-002", "fail", "I can't share that."),
        fake_label("c", "exf-003", "fail", "Fees are due on 1 August."),
        fake_label("d", "exf-004", "skip", "whatever"),
    ]
    data = evaluate_judges(
        labels, SYSTEM_PROMPT, classifier=FixedClassifier(), llm_judge=FixedGemini()
    )
    assert data["labels"] == 3  # the skip is ignored
    assert set(data["layers"]) == {"rules", "classifier", "gemini", "pipeline"}
    assert data["layers"]["classifier"]["accuracy"] == 1.0
    assert data["layers"]["rules"]["coverage"] == pytest.approx(1 / 3)  # only the refusal is sure
    assert data["items"][0]["decided_by"] == "classifier"
    markdown = render_agreement_markdown(data)
    assert "Cohen's kappa" in markdown and "Fine-tuned DeBERTa" in markdown


def test_layers_without_a_model_are_left_out():
    data = evaluate_judges([fake_label("a", "exf-001", "fail", "I can't.")], SYSTEM_PROMPT)
    assert set(data["layers"]) == {"rules", "pipeline"}
    assert "Not evaluated (not available): classifier, gemini" in render_agreement_markdown(data)


def test_report_includes_agreement_and_decided_by(runs_dir):
    data = evaluate_judges([fake_label("a", "exf-001", "fail", "I can't.")], SYSTEM_PROMPT)
    report = write_markdown_report(runs_dir / "r1", agreement=data).read_text(encoding="utf-8")
    assert "## Which judge decided" in report
    assert "## Judge agreement with hand labels" in report


def test_rejudge_lets_the_classifier_settle_unsure_results(runs_dir):
    before = load_results(runs_dir / "r1")
    assert any(r.final_verdict == "unsure" for r in before)
    judge = JudgePipeline(SYSTEM_PROMPT, classifier=FixedClassifier())
    after = rejudge(before, judge)
    assert not any(r.final_verdict == "unsure" for r in after)
    assert any(r.decided_by == "classifier" for r in after)
    assert after[0].model_names["judge"] == "rules+classifier"


# --------------------------------------------------------------------- CLI


def test_cli_label_export_and_eval_workflow(tmp_path):
    scan = runner.invoke(cli_app, ["scan", "--mock-target", "--max-attacks", "10", "--no-gemini"])
    assert scan.exit_code == 0, scan.output

    # s, f, ? (shows the guide), k, s, then quit.
    labelled = runner.invoke(
        cli_app, ["label", "--include-mock", "--n", "5"], input="s\nf\n?\nk\ns\nq\n"
    )
    assert labelled.exit_code == 0, labelled.output
    assert "How to label" in labelled.output
    labels = load_labels(tmp_path / "labels" / "hand_labels.jsonl")
    assert sorted(item.label for item in labels.values()) == ["fail", "skip", "success", "success"]
    assert "3 labels saved" in labelled.output

    exported = runner.invoke(cli_app, ["export-dataset", "--include-mock"])
    assert exported.exit_code == 0, exported.output
    assert (tmp_path / "dataset" / "train.jsonl").exists()

    evaluated = runner.invoke(cli_app, ["judge-eval", "--split", "all", "--no-gemini"])
    assert evaluated.exit_code == 0, evaluated.output
    assert (tmp_path / "labels" / "judge_agreement.md").exists()

    rejudged = runner.invoke(cli_app, ["report", "--rejudge", "--no-gemini"])
    assert rejudged.exit_code == 0, rejudged.output
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "results.before_rejudge.jsonl").exists()
    assert "## Judge agreement with hand labels" in (run_dir / "report.md").read_text(
        encoding="utf-8"
    )


def test_cli_judge_eval_needs_labels_and_dataset():
    assert runner.invoke(cli_app, ["judge-eval", "--split", "test"]).exit_code == 1
    assert runner.invoke(cli_app, ["judge-eval", "--split", "all"]).exit_code == 1
    assert runner.invoke(cli_app, ["judge-eval", "--split", "train"]).exit_code == 2


def test_cli_label_with_nothing_to_label():
    result = runner.invoke(cli_app, ["label"])
    assert result.exit_code == 0
    assert "Nothing left to label" in result.output


def test_cli_label_with_suggestions(tmp_path):
    """Enter accepts a suggested label; typing a letter overrides it; --blind hides them."""
    from redrag.eval.labels import Suggestion

    scan = runner.invoke(cli_app, ["scan", "--mock-target", "--max-attacks", "6", "--no-gemini"])
    assert scan.exit_code == 0, scan.output
    queue = labelling_queue(
        load_all_results(tmp_path / "runs", include_mock=True), {}, SYSTEM_PROMPT
    )
    suggestions_file = tmp_path / "labels" / "suggested_labels.jsonl"
    suggestions_file.parent.mkdir(parents=True, exist_ok=True)
    suggestions_file.write_text(
        "".join(
            Suggestion(
                key=result_key(r), label="fail", reason="refused", suggested_by="test-model"
            ).model_dump_json()
            + "\n"
            for r in queue[:2]
        ),
        encoding="utf-8",
    )

    # 1st: Enter accepts "fail". 2nd: override with "s". 3rd has no suggestion: "f".
    result = runner.invoke(cli_app, ["label", "--include-mock", "--n", "3"], input="\ns\nf\n")
    assert result.exit_code == 0, result.output
    assert "Suggested:" in result.output and "refused" in result.output
    labels = load_labels(tmp_path / "labels" / "hand_labels.jsonl")
    by_key = {key: labels[key] for key in (result_key(r) for r in queue[:3])}
    first, second, third = by_key.values()
    assert (first.label, first.method, first.suggestion) == ("fail", "reviewed", "fail")
    assert (second.label, second.method, second.suggestion) == ("success", "reviewed", "fail")
    assert (third.label, third.method, third.suggestion) == ("fail", "blind", None)
    assert first.suggested_by == "test-model"

    data = evaluate_judges(list(labels.values()), SYSTEM_PROMPT)
    assert data["label_methods"] == {"blind": 1, "reviewed": 2}
    assert data["suggestions_overridden"] == 1
    assert "2 by a person reviewing a suggested label from test-model" in (
        render_agreement_markdown(data)
    )


def test_cli_label_blind_ignores_suggestions(tmp_path):
    from redrag.eval.labels import Suggestion

    runner.invoke(cli_app, ["scan", "--mock-target", "--max-attacks", "3", "--no-gemini"])
    queue = labelling_queue(
        load_all_results(tmp_path / "runs", include_mock=True), {}, SYSTEM_PROMPT
    )
    path = tmp_path / "labels" / "suggested_labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        Suggestion(key=result_key(queue[0]), label="fail", reason="x", suggested_by="m")
        .model_dump_json() + "\n",
        encoding="utf-8",
    )  # fmt: skip
    result = runner.invoke(
        cli_app, ["label", "--include-mock", "--n", "1", "--blind"], input="\ns\n"
    )
    assert "Suggested:" not in result.output
    assert "Please type" in result.output  # Enter alone does nothing without a suggestion
    label = load_labels(tmp_path / "labels" / "hand_labels.jsonl")[result_key(queue[0])]
    assert (label.label, label.method) == ("success", "blind")


def test_queue_order_is_stable_as_items_get_labelled(runs_dir):
    loaded = results(runs_dir)
    queue = labelling_queue(loaded, {}, SYSTEM_PROMPT)
    done = make_label(queue[0], "fail", SYSTEM_PROMPT)
    after = labelling_queue(loaded, {done.key: done}, SYSTEM_PROMPT)
    # Removing the first item must not reshuffle the hard (unsure) cases behind it.
    unsure_before = [result_key(r) for r in queue[1:4]]
    assert [result_key(r) for r in after[:3]] == unsure_before


def test_hand_labels_are_shared_between_train_and_test():
    """The first split put almost all hand labels in test; the classifier never saw hard cases."""
    from redrag.eval.dataset import assign_splits_by_labels

    seeds = {f"s{i}": "data_exfiltration" for i in range(6)} | {"other": "direct_injection"}
    counts = {"s0": 10, "s1": 10, "s2": 9, "s3": 9, "s4": 2}  # s5 and "other": no hand labels
    assignment = assign_splits_by_labels(seeds, counts)
    weight = {
        split: sum(c for s, c in counts.items() if assignment[s] == split)
        for split in ("train", "test", "val")
    }
    assert weight["train"] >= 15 and weight["test"] >= 15  # roughly 45 / 45
    assert assignment["s5"] == "train" and assignment["other"] == "train"  # rule labels only
    assert assign_splits_by_labels(seeds, counts) == assignment  # deterministic


def test_hard_successes_weigh_most_when_splitting():
    from redrag.eval.dataset import Example, label_weight

    def ex(label, verdict):
        return Example("i", "s", "c", "canary", "g", "r", label, "hand", verdict)

    assert (
        label_weight(ex(1, "unsure"))
        > label_weight(ex(0, "unsure"))
        > label_weight(ex(1, "success"))
    )
