"""Tests for the CLI entry point, the client factory, and the repo layout."""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

from typer.testing import CliRunner

import redrag
from redrag.cli import app
from redrag.config import PROJECT_ROOT, load_config
from redrag.llm import MockClient, OllamaClient, build_gemini_client, build_local_client

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["scan", "report", "serve-target", "mcp", "show-config"]:
        assert command in result.output


def test_python_dash_m_entry_point_works():
    """The M0 acceptance check: `python -m redrag.cli --help`."""
    result = subprocess.run(
        [sys.executable, "-m", "redrag.cli", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "scan" in result.stdout


def test_unbuilt_command_exits_with_milestone_hint():
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 1
    assert "M6" in result.output


def test_scan_mock_target_then_report(tmp_path):
    """The M2 acceptance check in miniature: `redrag scan` writes results and a report."""
    result = runner.invoke(
        app,
        [
            "scan",
            "--mock-target",
            "--max-attacks",
            "5",
            "--defenses",
            "none",
            "--defenses",
            "all",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Attack success rate" in result.output

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "results.jsonl").exists()
    assert (run_dirs[0] / "report.md").exists()

    (run_dirs[0] / "report.md").unlink()
    again = runner.invoke(app, ["report", str(run_dirs[0])])
    assert again.exit_code == 0, again.output
    assert (run_dirs[0] / "report.md").exists()


def test_scan_rejects_bad_defense_and_category():
    assert runner.invoke(app, ["scan", "--mock-target", "-d", "firewall"]).exit_code == 2
    assert runner.invoke(app, ["scan", "--mock-target", "--categories", "nope"]).exit_code == 2
    assert runner.invoke(app, ["scan", "--mock-target", "--mutator", "magic"]).exit_code == 2


def test_agent_scan_on_mock_target(tmp_path):
    result = runner.invoke(
        app,
        [
            "scan",
            "--mock-target",
            "--agent",
            "--categories",
            "direct_injection",
            "--max-attacks",
            "5",
            "-d",
            "all",
            "--max-attempts",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "all + agent" in result.output
    assert "mutator template" in result.output  # mock target always uses templates
    report = next(tmp_path.iterdir()) / "report.md"
    assert "## Adaptive attacker" in report.read_text(encoding="utf-8")


def test_scan_explains_when_target_is_down(tmp_path):
    # Port 9 on localhost has nothing listening, so the connection is refused at once.
    result = runner.invoke(
        app,
        [
            "scan",
            "--target-url",
            "http://127.0.0.1:9",
            "--max-attacks",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "redrag serve-target" in result.output


def test_report_on_missing_run_fails(tmp_path):
    assert runner.invoke(app, ["report", str(tmp_path / "nope")]).exit_code == 1


def test_show_config_hides_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-print")
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "should-not-print" not in result.output
    assert "**********" in result.output


def test_factory_returns_mock_or_ollama():
    mock_config = load_config(env={"REDRAG_LLM_PROVIDER": "mock"})
    assert isinstance(build_local_client(mock_config, "victim"), MockClient)

    ollama_config = load_config(env={"OLLAMA_ATTACKER_MODEL": "qwen2.5:3b"})
    attacker = build_local_client(ollama_config, "attacker")
    assert isinstance(attacker, OllamaClient)
    assert attacker.model == "qwen2.5:3b"


def test_gemini_factory_does_not_need_network_or_key():
    client = build_gemini_client(load_config(env={}))
    assert client.model == "gemini-2.5-flash-lite"


def test_every_module_imports():
    """All placeholder modules must at least import cleanly."""
    for module in pkgutil.walk_packages(redrag.__path__, prefix="redrag."):
        importlib.import_module(module.name)


def test_expected_layout_exists():
    expected = [
        "README.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "config/default.yaml",
        "redrag/target/data/handbook",
        "redrag/attacks/seeds",
        "redrag/eval/benign_questions.jsonl",
        "notebooks/finetune_judge.ipynb",
        "dashboard/app.py",
        ".github/workflows/redteam.yml",
    ]
    missing = [path for path in expected if not (PROJECT_ROOT / path).exists()]
    assert missing == []


def test_gitignore_protects_secrets_and_spec():
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in [".env", "spec.md", "data/chroma/", "*.pt", "*.bin"]:
        assert entry in ignored
