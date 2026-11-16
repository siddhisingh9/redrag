"""MCP server: lets an AI assistant run RedRAG scans as a tool.

MCP (Model Context Protocol) is an open standard for connecting AI apps
(Claude Desktop, Claude Code, VS Code, Cursor, ...) to outside tools. The app
starts `redrag mcp` as a subprocess and talks JSON-RPC to it over stdin/stdout
(the "stdio transport"). Then you can ask the assistant things like "scan my RAG
app at localhost:8000 for prompt injection" and it calls `scan_rag_app` itself.

Three tools (spec 6.8):
- `scan_rag_app(target_url, categories?, defenses?, max_attacks?)`: run a scan,
  get the headline numbers back plus the paths of the saved reports.
- `list_attacks(category?)`: the seed attacks RedRAG can run.
- `get_report(run_id)`: the summary and Markdown report of an earlier scan.

Design notes:
- Built with `MCPServer` from the official MCP Python SDK. (SDK v2 renamed
  `FastMCP` to `MCPServer`; it works the same way: type hints become the tool's
  input schema, docstrings become its description.)
- The tools are plain sync functions. The SDK runs them in a worker thread, so a
  scan that takes minutes doesn't freeze the server.
- In stdio mode, stdout *is* the protocol channel. The SDK points stray stdout
  writes at stderr while serving, and nothing here prints.
- Failures we can foresee (target not running, unknown run id, bad category)
  raise `ToolError`, so the assistant gets a readable message it can act on
  instead of a generic "Error executing tool".
- Same scan code as `redrag scan` (runner + `redrag.factory`), so results and
  reports are identical to the CLI's.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from redrag.config import Config, load_config, resolve_path

MOCK_TARGET = "mock"
# Each attack against a real 3B model takes a few seconds per defense
# configuration. 10 attacks x 2 configurations is about 1-3 minutes: long enough
# to be meaningful, short enough that the MCP client doesn't time out.
DEFAULT_MAX_ATTACKS = 10
MAX_LISTED_SUCCESSES = 20
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

INSTRUCTIONS = """RedRAG red-teams RAG (retrieval-augmented generation) chat apps.
Use scan_rag_app to attack a target app with benign test attacks (marker words, fake
canary secrets, system-prompt leaks) and measure the attack success rate (ASR) with
and without defenses. Use target_url="mock" for an offline demo target. Use
list_attacks to see what can be run and get_report to read an earlier scan."""


# --------------------------------------------------------------------- tools


def scan_rag_app(
    target_url: Annotated[
        str,
        Field(
            description='Base URL of the target RAG app, e.g. "http://127.0.0.1:8000" '
            '(start it with `redrag serve-target`). Use "mock" for an offline in-process '
            "demo target."
        ),
    ],
    categories: Annotated[
        list[str] | None,
        Field(
            description="Attack categories to run (default: all). Choose from: "
            "direct_injection, indirect_injection, system_prompt_extraction, "
            "data_exfiltration, multi_turn_escalation."
        ),
    ] = None,
    defenses: Annotated[
        list[str] | None,
        Field(
            description='Defense configurations to compare, e.g. ["none", "all"] (default) '
            'or ["none", "input_filter,output_leak_filter"]. Defenses: input_filter, '
            "context_sanitiser, output_leak_filter, judge_guard."
        ),
    ] = None,
    max_attacks: Annotated[
        int | None,
        Field(
            ge=1,
            description=f"Run at most this many attacks, spread across categories "
            f"(default {DEFAULT_MAX_ATTACKS}). Each attack runs once per defense configuration.",
        ),
    ] = DEFAULT_MAX_ATTACKS,
    include_benign: Annotated[
        bool,
        Field(
            description="Also ask 50 harmless questions per configuration to measure the "
            "false-refusal rate (slower)."
        ),
    ] = False,
) -> dict[str, Any]:
    """Red-team a RAG app: run benign test attacks and report how many got through.

    Returns the attack success rate (ASR) for each defense configuration, the
    attacks that succeeded, and paths to the full Markdown and HTML reports.
    """
    import httpx

    from redrag.attacks.loader import AttackFileError, load_attacks, select_attacks
    from redrag.eval.benign import load_benign_questions
    from redrag.factory import mock_target_client, write_reports
    from redrag.runner import TargetClient, TargetError, normalise_defense_config, run_scan

    config = load_config()
    try:
        configs = [normalise_defense_config(value) for value in (defenses or ["none", "all"])]
        wanted = categories or config.attacks.categories
        attacks = select_attacks(load_attacks(categories=[c.strip() for c in wanted]), max_attacks)
    except (ValueError, AttackFileError) as exc:
        raise ToolError(str(exc)) from exc
    if not attacks:
        raise ToolError("No attacks match those categories.")

    if target_url.strip().lower() == MOCK_TARGET:
        http, url = mock_target_client(config), "in-process mock target"
    else:
        url = _check_url(target_url)
        http = httpx.Client(base_url=url, timeout=300)

    try:
        target = TargetClient(http)
        target.health()  # fail fast, before loading the judge model
        summary = run_scan(
            attacks,
            target,
            _judge(config),
            configs,
            runs_dir=resolve_path(config.output.runs_dir),
            target_url=url,
            benign_questions=load_benign_questions() if include_benign else [],
        )
    except TargetError as exc:
        raise ToolError(
            f"{exc} Is the target running? Start the demo target with `redrag serve-target`, "
            'or pass target_url="mock".'
        ) from exc
    finally:
        http.close()

    write_reports(summary.run_dir, config)
    return run_summary(summary.run_dir)


def list_attacks(
    category: Annotated[
        str | None,
        Field(description="Only this category (default: all five)."),
    ] = None,
) -> dict[str, Any]:
    """List the seed attacks RedRAG can run, with their messages and success signal."""
    from redrag.attacks.loader import AttackFileError, load_attacks

    try:
        attacks = load_attacks(categories=[category] if category else None)
    except AttackFileError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "count": len(attacks),
        "attacks": [
            {
                "id": attack.id,
                "category": attack.category,
                "owasp": attack.owasp,
                "language": attack.language,
                "turns": attack.turns,
                "poison_doc": attack.poison_doc.text if attack.poison_doc else None,
                "success_signal": attack.success_signal.model_dump(exclude_none=True),
                "notes": attack.notes,
            }
            for attack in attacks
        ],
    }


def get_report(
    run_id: Annotated[
        str,
        Field(description='The run id returned by scan_rag_app, or "latest" (default).'),
    ] = "latest",
) -> dict[str, Any]:
    """Get the summary and full Markdown report of an earlier scan."""
    from redrag.eval.report import REPORT_FILE
    from redrag.factory import write_reports

    config = load_config()
    run_dir = find_run(run_id, config)
    report_path = run_dir / REPORT_FILE
    if not report_path.exists():  # e.g. a scan that was interrupted before the report
        write_reports(run_dir, config)
    summary = run_summary(run_dir)
    summary["report"] = report_path.read_text(encoding="utf-8")
    return summary


# ------------------------------------------------------------------ helpers

_judges: dict[tuple, Any] = {}


def _judge(config: Config):
    """The judge pipeline, built once per server process.

    Loading the DeBERTa classifier takes a few seconds; the MCP server is
    long-lived, so we keep it instead of reloading it on every scan. The key
    changes if the model folder or the Gemini settings do.
    """
    from redrag.factory import build_judge

    key = (
        config.judge.classifier_path,
        config.judge.use_gemini,
        config.gemini.api_key is not None,
    )
    if key not in _judges:
        _judges[key] = build_judge(config)
    return _judges[key]


def _check_url(target_url: str) -> str:
    """Accept only http(s) URLs, so the tool can't be pointed at file:// or other schemes."""
    import httpx

    try:
        url = httpx.URL(target_url.strip())
    except httpx.InvalidURL as exc:
        raise ToolError(f"Not a valid URL: {target_url!r}") from exc
    if url.scheme not in ("http", "https") or not url.host:
        raise ToolError(
            f'target_url must be an http(s) URL such as "http://127.0.0.1:8000" '
            f'(or "mock"), got {target_url!r}'
        )
    return str(url).rstrip("/")


def find_run(run_id: str, config: Config) -> Path:
    """Resolve a run id to its folder inside the runs directory.

    The id comes from the AI client, so it is checked: plain folder names only,
    never a path. That way `get_report("../../somewhere")` can't read files
    outside the runs folder.
    """
    from redrag.factory import latest_run_dir

    runs_dir = resolve_path(config.output.runs_dir)
    if run_id == "latest":
        run_dir = latest_run_dir(runs_dir)
        if run_dir is None:
            raise ToolError("No scans yet. Run scan_rag_app first.")
        return run_dir
    if not RUN_ID_PATTERN.match(run_id) or ".." in run_id:
        raise ToolError(f"Invalid run id {run_id!r}: expected a folder name like 20261116-034159.")
    run_dir = runs_dir / run_id
    if not (run_dir / "results.jsonl").exists():
        raise ToolError(f"No scan called {run_id!r} in {runs_dir}.")
    return run_dir


def run_summary(run_dir: Path) -> dict[str, Any]:
    """The headline numbers of a run, as plain JSON for the AI client."""
    from redrag.eval.benign import load_benign_results
    from redrag.eval.html_report import HTML_REPORT_FILE
    from redrag.eval.metrics import best_per_seed
    from redrag.eval.report import (
        REPORT_FILE,
        best_tradeoff,
        defense_summary,
        load_results,
        load_run_info,
    )

    info = load_run_info(run_dir)
    results = load_results(run_dir)
    benign = load_benign_results(run_dir)
    agent_mode = any(result.attempt > 0 for result in results)
    rows = defense_summary(results, benign, agent_mode)
    best = best_tradeoff(rows)

    successes = [
        result
        for result in (best_per_seed(results) if agent_mode else results)
        if result.final_verdict == "success"
    ]
    configurations = []
    for row in rows:
        entry: dict[str, Any] = {
            "configuration": row.config,
            "defenses": row.defenses,
            "attack_success_rate": _round(row.asr.asr),
            "attacks_succeeded": row.asr.successes,
            "attacks_judged": row.asr.judged,
            "unsure": row.asr.unsure,
            "errors": row.asr.errors,
        }
        if row.frr is not None:
            entry["false_refusal_rate"] = _round(row.frr.frr)
            entry["useful_answer_rate"] = _round(row.frr.usefulness)
        configurations.append(entry)

    return {
        "run_id": run_dir.name,
        "target_url": info.get("target_url"),
        "target_model": info.get("target_model"),
        "judge_layers": info.get("judge_layers"),
        "configurations": configurations,
        "best_balance": best.config if best else None,
        "successful_attacks": [
            {
                "attack_id": result.attack_id,
                "category": result.category,
                "configuration": result.defense_config,
                "decided_by": result.decided_by,
            }
            for result in successes[:MAX_LISTED_SUCCESSES]
        ],
        "successful_attacks_total": len(successes),
        "report_markdown": str(run_dir / REPORT_FILE),
        "report_html": str(run_dir / HTML_REPORT_FILE),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


# ------------------------------------------------------------------- server


def build_server() -> MCPServer:
    """The RedRAG MCP server with its three tools registered."""
    server = MCPServer(name="redrag", instructions=INSTRUCTIONS, log_level="WARNING")
    server.tool()(scan_rag_app)
    server.tool()(list_attacks)
    server.tool()(get_report)
    return server


def serve_stdio() -> None:
    """Run the server over stdin/stdout until the client disconnects (`redrag mcp`)."""
    build_server().run("stdio")
