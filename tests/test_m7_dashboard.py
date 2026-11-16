"""Tests for M7: the Streamlit dashboard (data layer + every page, headless)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from dashboard import data
from redrag.agent.attacker_graph import run_agent_scan
from redrag.agent.mutators import TemplateMutator
from redrag.attacks.loader import load_attacks, select_attacks
from redrag.config import PROJECT_ROOT, load_config
from redrag.eval.benign import load_benign_questions
from redrag.judge.pipeline import JudgePipeline
from redrag.llm import MockClient
from redrag.runner import TargetClient, run_scan
from redrag.target.app import create_app
from redrag.target.rag import SYSTEM_PROMPT

APP = str(PROJECT_ROOT / "dashboard" / "app.py")


@pytest.fixture
def runs(pipeline, tmp_path):
    """Two saved runs in the test runs folder: a plain scan (with FRR) and an agent scan."""
    runs_dir = tmp_path / "runs"  # conftest points REDRAG_RUNS_DIR here
    app = create_app(load_config(env={}), MockClient(model="mock-victim"), pipeline)
    target = TargetClient(TestClient(app))
    attacks = select_attacks(load_attacks(), 10)
    run_scan(
        attacks, target, JudgePipeline(SYSTEM_PROMPT), ["none", "input_filter", "all"],
        runs_dir=runs_dir, target_url="test", run_id="20260101-000000",
        benign_questions=load_benign_questions()[:10],
    )  # fmt: skip
    run_agent_scan(
        attacks, target, JudgePipeline(SYSTEM_PROMPT), ["all"],
        runs_dir=runs_dir, target_url="test", mutator=TemplateMutator(), max_attempts=3,
        run_id="20260102-000000",
    )  # fmt: skip
    return runs_dir


def app_test() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


# ----------------------------------------------------------------- data layer


def test_list_runs_newest_first(runs):
    found = data.list_runs(runs)
    assert [r.run_id for r in found] == ["20260102-000000", "20260101-000000"]
    assert [r.mode for r in found] == ["agent", "scan"]
    assert found[1].has_benign
    assert "3 configurations" in found[1].label
    assert data.list_runs(runs / "missing") == []


def test_overview_tables(runs):
    run = data.load_run(runs / "20260101-000000")
    summary = data.summary_frame(run)
    assert list(summary["Configuration"]) == ["none", "input_filter", "all"]
    assert summary["ASR"].iloc[0] > summary["ASR"].iloc[-1]  # defenses help on the mock
    assert summary["FRR"].notna().all()
    headline = data.headline(run)
    assert headline["asr_none"] == summary["ASR"].iloc[0]
    cells = data.category_frame(run)
    assert set(cells["Category"]) == {data.category_name(c) for c in data.CATEGORIES}
    assert cells["ASR"].between(0, 1).all()


def test_attack_table_and_filters(runs):
    run = data.load_run(runs / "20260101-000000")
    frame = data.attacks_frame(run.results)
    assert len(frame) == len(run.results) == 30
    only = data.filter_attacks(frame, configs=["all"], verdicts=["success"])
    assert set(only["Configuration"]) <= {"all"} and set(only["Verdict"]) <= {"success"}
    assert len(data.filter_attacks(frame, search="DIR-001")) == 3  # case-insensitive id match
    assert data.filter_attacks(frame).equals(frame)  # no filters = everything
    blocked = frame[frame["Blocked by"] != ""]
    assert set(blocked["Configuration"]) <= {"input_filter", "all"}


def test_lineage(runs):
    run = data.load_run(runs / "20260102-000000")
    assert run.agent_mode
    outcomes = data.seed_outcomes(run.results, "all")
    assert len(outcomes) == 10
    assert set(outcomes["Outcome"]) <= {
        "worked as written",
        "rescued by the agent",
        "never worked",
    }
    seed = outcomes[outcomes["Attempts"] > 1]["Seed"].iloc[0]
    nodes = data.lineage(run.results, "all", seed)
    assert nodes[0].result.attempt == 0 and nodes[0].parent is None
    assert [n.result.attempt for n in nodes] == sorted(n.result.attempt for n in nodes)
    assert all(n.parent for n in nodes[1:])
    dot = data.lineage_dot(nodes, {"success": "#000", "fail": "#eee"})
    assert dot.startswith("digraph") and f'"{seed}" ->' in dot
    assert "SUCCESS" in dot or "FAIL" in dot  # verdict in words, not colour alone
    strategies = data.strategy_frame(run.results, "all")
    assert strategies["Variants"].sum() == sum(r.attempt > 0 for r in run.results)


def test_dot_escapes_quotes():
    assert data._dot_escape('a "b"\nc') == 'a \\"b\\" c'


# ------------------------------------------------------------------- pages


def test_every_page_renders_on_a_saved_scan(runs):
    """The M7 acceptance check: the dashboard runs on a saved scan."""
    at = app_test().run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Overview"
    # The newest run (the agent scan) is selected first; switch to the plain scan.
    assert at.selectbox(key="run_id").value == "20260102-000000"
    at.selectbox(key="run_id").set_value("20260101-000000").run()
    tiles = {m.label: m.value for m in at.metric}
    assert {"ASR, no defenses", "ASR, all defenses", "FRR, all defenses", "Best balance"} <= set(
        tiles
    )

    for page in ["views/attacks.py", "views/lineage.py", "views/reports.py", "views/label.py"]:
        at.switch_page(page).run()
        assert not at.exception, (page, at.exception)


def test_overview_of_an_agent_run(runs):
    at = app_test()
    at.session_state["run_id"] = "20260102-000000"
    at.run()
    assert not at.exception, at.exception
    tiles = {m.label: m.value for m in at.metric}
    assert "Seeds rescued by the agent" in tiles and "ASR after the agent" in tiles


def test_run_can_be_opened_by_link(runs):
    at = app_test()
    at.query_params["run"] = "20260101-000000"
    at.run()
    assert at.selectbox(key="run_id").value == "20260101-000000"
    assert {m.label for m in at.metric} >= {"FRR, all defenses"}


def test_lineage_page_points_plain_scans_to_agent_runs(runs):
    at = app_test()
    at.session_state["run_id"] = "20260101-000000"
    at.run()
    at.switch_page("views/lineage.py").run()
    assert "no rewritten variants" in at.info[0].value
    at.button(key="open-20260102-000000").click().run()
    assert at.selectbox(key="run_id").value == "20260102-000000"
    assert not at.exception, at.exception
    assert any(m.label == "Rescued by the agent" for m in at.metric)


def test_empty_runs_folder_shows_a_hint(monkeypatch):
    monkeypatch.setenv("REDRAG_DEMO", "0")  # otherwise the bundled demo scans are shown
    at = app_test().run()
    assert not at.exception, at.exception
    assert "No saved scans yet" in at.info[0].value


def test_label_page_saves_a_label(runs, tmp_path):
    labels_file = tmp_path / "labels" / "hand_labels.jsonl"  # conftest's labels path
    at = app_test().run()
    at.switch_page("views/label.py").run()
    assert not at.exception, at.exception
    assert "Nothing left to label" in at.success[0].value  # mock runs are skipped by default
    at.toggle(key="include_mock").set_value(True).run()
    assert "0 of 150 labels" in at.get("progress")[0].proto.text
    at.button(key="fail").click().run()
    saved = [json.loads(line) for line in labels_file.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 1
    assert saved[0]["label"] == "fail" and saved[0]["method"] == "blind"  # no suggestions file
    assert "1 of 150 labels" in at.get("progress")[0].proto.text


def test_theme_palettes_have_the_length_streamlit_requires():
    """Streamlit silently ignores a sequential palette that isn't exactly 10 colours."""
    import re

    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    ramps = re.findall(r"chartSequentialColors = \[([^\]]*)\]", config)
    assert len(ramps) == 2  # light and dark
    assert all(len(re.findall(r"#[0-9a-f]{6}", ramp)) == 10 for ramp in ramps)


# ------------------------------------------------------------------- demo


def test_demo_mode_shows_bundled_scans_read_only():
    """With no scans of your own (fresh clone, hosted demo), the example scans are shown."""
    at = app_test().run()  # conftest's runs folder is empty, so demo mode switches on
    assert not at.exception, at.exception
    picker = at.selectbox(key="run_id")
    assert picker.value == "20261116-034159"  # demo/runs/index.json puts the sweep first
    assert "Every defense combination" in picker.format_func(picker.value)
    assert {m.label for m in at.metric} >= {"ASR, no defenses", "FRR, all defenses"}
    # The Label page writes files, so the read-only demo doesn't register it.
    with pytest.raises(ValueError, match="Could not find a navigation page"):
        at.switch_page("views/label.py").run()
    for page in ["views/attacks.py", "views/lineage.py", "views/reports.py"]:
        at.switch_page(page).run()
        assert not at.exception, (page, at.exception)

    titles = [run.title for run in data.list_runs(PROJECT_ROOT / "demo" / "runs")]
    assert all(titles) and len(titles) == 3


def test_demo_runs_without_the_heavy_dependencies():
    """The hosted demo installs only dashboard/requirements.txt: no PyTorch, Chroma,
    Ollama/Gemini clients, FastAPI or Phoenix. Run every page with those blocked."""
    import subprocess
    import sys

    script = """
import importlib.abc, sys
HEAVY = ("torch", "transformers", "sentence_transformers", "chromadb", "fastapi", "uvicorn",
         "langgraph", "google.genai", "phoenix", "opentelemetry.sdk", "datasets", "peft",
         "mcp", "httpx", "typer")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == h or name.startswith(h + ".") for h in HEAVY):
            raise ImportError("not in dashboard/requirements.txt: " + name)
sys.meta_path.insert(0, Block())
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=60).run()
assert not at.exception, at.exception
for page in ["views/attacks.py", "views/lineage.py", "views/reports.py"]:
    at.switch_page(page).run()
    assert not at.exception, (page, at.exception)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, APP],
        cwd=PROJECT_ROOT,
        env={**__import__("os").environ, "REDRAG_DEMO": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_dashboard_requirements_are_pinned():
    lines = (PROJECT_ROOT / "dashboard" / "requirements.txt").read_text(encoding="utf-8")
    packages = [line for line in lines.splitlines() if line and not line.startswith("#")]
    assert packages and all("==" in line for line in packages)
    assert not any(line.split("==")[0] in ("torch", "chromadb") for line in packages)


def test_report_page_shows_and_offers_the_full_report(runs):
    """Saved report.html is shown as-is; a scan without one gets it built on the fly."""
    at = app_test()
    at.session_state["run_id"] = "20260101-000000"  # the fixture writes no report.html
    at.run()
    at.switch_page("views/reports.py").run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Report"
    download = at.get("download_button")[0]
    assert download.proto.label == "Download HTML"
