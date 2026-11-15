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
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "M2" in result.output


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
