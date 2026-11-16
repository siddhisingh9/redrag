"""Tests for M6: the MCP server and the CI red-team gate."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import pytest
import yaml
from typer.testing import CliRunner

from redrag.attacks.schema import AttackResult
from redrag.cli import app as cli_app
from redrag.config import PROJECT_ROOT
from redrag.eval.ci_gate import Baseline, check, load_baseline
from redrag.mcp_server import build_server

runner = CliRunner()


def call(tool: str, arguments: dict | None = None):
    """Call one tool through a real MCP client connected in-process to the server."""
    from mcp import Client

    async def go():
        async with Client(build_server()) as client:
            return await client.call_tool(tool, arguments or {})

    return asyncio.run(go())


# ------------------------------------------------------------------ MCP server


def test_server_lists_the_three_tools():
    from mcp import Client

    async def go():
        async with Client(build_server()) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == {"scan_rag_app", "list_attacks", "get_report"}
    scan = tools["scan_rag_app"].input_schema
    assert scan["required"] == ["target_url"]  # everything else is optional (spec 6.8)
    assert {"categories", "defenses", "max_attacks"} <= set(scan["properties"])
    assert "description" in scan["properties"]["target_url"]


def test_list_attacks():
    everything = call("list_attacks").structured_content
    assert everything["count"] == 55
    direct = call("list_attacks", {"category": "direct_injection"}).structured_content
    assert direct["count"] == 11
    assert all(a["category"] == "direct_injection" for a in direct["attacks"])
    assert direct["attacks"][0]["success_signal"] == {"type": "marker", "value": "PINEAPPLE-7"}
    indirect = call("list_attacks", {"category": "indirect_injection"}).structured_content
    assert all(a["poison_doc"] for a in indirect["attacks"])

    bad = call("list_attacks", {"category": "nope"})
    assert bad.is_error and "Unknown categories" in bad.content[0].text


def test_scan_then_get_report(tmp_path):
    result = call("scan_rag_app", {"target_url": "mock", "max_attacks": 5})
    assert not result.is_error, result.content
    summary = result.structured_content
    configs = {c["configuration"]: c for c in summary["configurations"]}
    assert set(configs) == {"none", "all"}
    assert configs["none"]["attacks_judged"] == 5
    assert configs["none"]["attack_success_rate"] > configs["all"]["attack_success_rate"]
    assert summary["judge_layers"] == ["rules"]  # no trained model in the test env
    assert summary["successful_attacks_total"] >= len(summary["successful_attacks"]) > 0
    run_dir = tmp_path / "runs" / summary["run_id"]  # conftest points runs_dir here
    assert (run_dir / "report.md").exists() and (run_dir / "report.html").exists()

    # By id, and "latest" (the default) finds the same run.
    report = call("get_report", {"run_id": summary["run_id"]}).structured_content
    assert report["report"].startswith("# RedRAG scan report")
    assert report["configurations"] == summary["configurations"]
    assert call("get_report").structured_content["run_id"] == summary["run_id"]

    # An interrupted scan has results but no report: get_report rebuilds it.
    (run_dir / "report.md").unlink()
    assert "report" in call("get_report", {"run_id": summary["run_id"]}).structured_content


def test_scan_options_and_benign_questions():
    result = call(
        "scan_rag_app",
        {
            "target_url": "MOCK",
            "categories": ["data_exfiltration"],
            "defenses": ["output_leak_filter"],
            "max_attacks": 2,
            "include_benign": True,
        },
    ).structured_content
    [config] = result["configurations"]
    assert config["configuration"] == "output_leak_filter"
    assert config["attacks_judged"] == 2
    assert config["false_refusal_rate"] is not None and "useful_answer_rate" in config


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"target_url": "file:///etc/passwd"}, "must be an http(s) URL"),
        ({"target_url": "not a url at all"}, "http(s) URL"),
        ({"target_url": "http://127.0.0.1:9", "max_attacks": 1}, "Is the target running?"),
        ({"target_url": "mock", "defenses": ["firewall"]}, "firewall"),
        ({"target_url": "mock", "categories": ["nope"]}, "Unknown categories"),
    ],
)
def test_scan_errors_are_readable(arguments, message):
    result = call("scan_rag_app", arguments)
    assert result.is_error
    assert message in result.content[0].text


def test_get_report_rejects_paths_and_unknown_runs():
    assert "No scans yet" in call("get_report").content[0].text
    for bad in ["../../secrets", "..", "a/b", "C:\\Windows", ".hidden"]:
        result = call("get_report", {"run_id": bad})
        assert result.is_error and "Invalid run id" in result.content[0].text, bad
    assert "No scan called" in call("get_report", {"run_id": "20990101-000000"}).content[0].text


def test_mcp_over_stdio_subprocess(tmp_path):
    """The M6 acceptance check: a real client starts `redrag mcp` and runs a scan over stdio."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "redrag.cli", "mcp"],
        cwd=str(PROJECT_ROOT),
        env={
            "REDRAG_RUNS_DIR": str(tmp_path / "runs"),
            "REDRAG_JUDGE_MODEL_DIR": str(tmp_path / "no-model"),
            "REDRAG_LLM_PROVIDER": "mock",
            "PYTHONUTF8": "1",
        },
    )

    async def go():
        async with Client(params, read_timeout_seconds=120) as client:
            names = [tool.name for tool in (await client.list_tools()).tools]
            scan = await client.call_tool("scan_rag_app", {"target_url": "mock", "max_attacks": 2})
            return names, scan

    names, scan = asyncio.run(go())
    assert sorted(names) == ["get_report", "list_attacks", "scan_rag_app"]
    assert not scan.is_error, scan.content
    assert (tmp_path / "runs" / scan.structured_content["run_id"] / "report.html").exists()


# ------------------------------------------------------------------- CI gate


def _result(config: str, verdict: str, defenses: list[str] | None = None) -> AttackResult:
    return AttackResult(
        run_id="ci",
        attack_id="dir-001",
        category="direct_injection",
        owasp="LLM01",
        language="en",
        defense_config=config,
        defenses=defenses if defenses is not None else [],
        transcript=[],
        retrieved_doc_ids=[],
        judge_verdicts=[],
        final_verdict=verdict,
        decided_by="rules",
        latency_ms=0,
        model_names={},
        timestamp=datetime.now(timezone.utc),
    )


THREE = ["context_sanitiser", "input_filter", "output_leak_filter"]


def _scan(none_successes: int, all_successes: int, total: int = 10, defenses=THREE):
    def side(config, successes, active):
        return [
            _result(config, "success" if i < successes else "fail", active) for i in range(total)
        ]

    return side("none", none_successes, []) + side("all", all_successes, defenses)


BASELINE = Baseline(
    max_asr_all_defenses=0.25,
    min_asr_no_defenses=0.7,
    expected_defenses=THREE,
    recorded={"asr_all_defenses": 0.2},
)


def test_gate_passes_at_baseline():
    result = check(_scan(8, 2), BASELINE)
    assert result.passed and result.problems == [] and result.notes == []


def test_gate_fails_when_defenses_get_weaker():
    result = check(_scan(8, 3), BASELINE)  # 30% > 25%
    assert not result.passed
    assert "Regression" in result.problems[0]


def test_gate_fails_when_the_scanner_itself_breaks():
    # A judge that never says "success" would pass the ASR check forever.
    result = check(_scan(0, 0), BASELINE)
    assert not result.passed and "look broken" in result.problems[0]


def test_gate_fails_on_errors_and_mismatched_defenses():
    results = _scan(8, 2)
    results[-1] = _result("all", "error", THREE)
    assert "error" in check(results, BASELINE).problems[0]
    with_guard = check(_scan(8, 0, defenses=[*THREE, "judge_guard"]), BASELINE)
    assert not with_guard.passed and "--no-guard" in with_guard.problems[0]
    assert not check(_scan(8, 2)[:10], BASELINE).passed  # no "all" configuration


def test_gate_suggests_tightening_after_an_improvement():
    result = check(_scan(8, 1), BASELINE)
    assert result.passed and "lowering max_asr_all_defenses" in result.notes[0]


def test_committed_baseline_matches_the_mock_scan():
    """The numbers in config/ci_baseline.json are what the mock scan really gives."""
    baseline = load_baseline(PROJECT_ROOT / "config" / "ci_baseline.json")
    assert baseline.expected_defenses == THREE
    assert baseline.recorded["asr_all_defenses"] < baseline.max_asr_all_defenses
    # One more success than recorded (14/55) must already fail.
    assert 14 / 55 > baseline.max_asr_all_defenses >= 13 / 55


def test_ci_check_command_passes_on_mock_target(tmp_path):
    result = runner.invoke(cli_app, ["ci-check", "--mock-target", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output and "46/55" in result.output and "13/55" in result.output
    assert (next(tmp_path.iterdir()) / "report.html").exists()


def test_ci_check_catches_a_weakened_defense(tmp_path, monkeypatch):
    """Simulate a bad PR: the output leak filter stops blocking anything."""
    from redrag.target import defenses

    monkeypatch.setattr(
        defenses, "check_output", lambda *args, **kwargs: defenses.FilterResult(blocked=False)
    )
    result = runner.invoke(cli_app, ["ci-check", "--mock-target", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output and "Regression" in result.output


def test_ci_check_uses_given_baseline(tmp_path):
    strict = tmp_path / "strict.json"
    data = json.loads((PROJECT_ROOT / "config" / "ci_baseline.json").read_text(encoding="utf-8"))
    data["max_asr_all_defenses"] = 0.1
    data["scan"]["max_attacks"] = 10
    strict.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(
        cli_app,
        ["ci-check", "--mock-target", "--baseline", str(strict), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1 and "10 attacks" in result.output


def test_serve_target_no_guard(monkeypatch, tiny_judge_model):
    """--no-guard keeps judge_guard out, so CI's "all" means the same three defenses."""
    import uvicorn

    import redrag.target.app as target_app

    captured = {}
    monkeypatch.setenv("REDRAG_JUDGE_MODEL_DIR", str(tiny_judge_model))
    monkeypatch.setattr(target_app, "create_app", lambda config, guard=None: captured.update(
        guard=guard))  # fmt: skip
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

    assert runner.invoke(cli_app, ["serve-target", "--mock"]).exit_code == 0
    assert captured["guard"] is not None  # a model exists, so the guard is offered
    assert runner.invoke(cli_app, ["serve-target", "--mock", "--no-guard"]).exit_code == 0
    assert captured["guard"] is None


def test_workflow_runs_the_gate_on_pull_requests():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github" / "workflows" / "redteam.yml").read_text(encoding="utf-8")
    )
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert "pull_request" in triggers
    steps = workflow["jobs"]["redteam"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    for needed in ["ruff check", "ruff format --check", "pytest", "--index-url",
                   "serve-target --mock", "--no-guard", "redrag ci-check"]:  # fmt: skip
        assert needed in script, needed
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["redteam"]["env"]["REDRAG_LLM_PROVIDER"] == "mock"
