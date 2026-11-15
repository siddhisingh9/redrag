"""Command-line interface: `redrag scan | report | serve-target | mcp`.

Run it either way:
    python -m redrag.cli --help
    redrag --help            (after `pip install -e .`)

Built with Typer (turns typed Python functions into commands) and Rich (pretty
terminal output). `mcp` is still a placeholder (M6).

Heavy imports (FastAPI, Chroma, the runner) happen inside the commands, so
`redrag --help` stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from redrag.config import Config, load_config, resolve_path

app = typer.Typer(
    name="redrag",
    help="RedRAG: automated red-teaming for RAG applications.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _not_built_yet(command: str, milestone: str) -> None:
    """Tell the user a command is planned but not implemented yet, and exit with an error."""
    console.print(f"[yellow]`redrag {command}` is not built yet; it arrives in {milestone}.[/]")
    raise typer.Exit(code=1)


@app.command()
def scan(
    defenses: list[str] = typer.Option(
        ["none", "all"],
        "--defenses",
        "-d",
        help="Defense configuration to test: 'none', 'all', or a list such as "
        "'input_filter,output_leak_filter'. Repeat the option to compare several "
        "(default: none and all).",
    ),
    categories: str | None = typer.Option(
        None, help="Comma-separated attack categories. Default: attacks.categories in config."
    ),
    max_attacks: int | None = typer.Option(
        None, min=1, help="Run at most this many attacks, spread evenly across categories."
    ),
    target_url: str | None = typer.Option(
        None, help="Target app URL. Default: target.url in config."
    ),
    mock_target: bool = typer.Option(
        False,
        "--mock-target",
        help="Scan an in-process target with the MockClient and offline embeddings "
        "(no server, no Ollama). Good for a quick try-out.",
    ),
    output_dir: Path | None = typer.Option(
        None, help="Folder for run results. Default: output.runs_dir in config."
    ),
) -> None:
    """Run the attack suite against the target RAG app and save results + a report."""
    import httpx
    from rich.progress import Progress

    from redrag.attacks.loader import AttackFileError, load_attacks, select_attacks
    from redrag.eval.report import write_markdown_report
    from redrag.judge.pipeline import JudgePipeline
    from redrag.runner import TargetClient, TargetError, normalise_defense_config, run_scan
    from redrag.target.rag import SYSTEM_PROMPT

    config = load_config()

    try:
        configs = [normalise_defense_config(value) for value in defenses]
        wanted = categories.split(",") if categories else config.attacks.categories
        attacks = select_attacks(load_attacks(categories=[c.strip() for c in wanted]), max_attacks)
    except (ValueError, AttackFileError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    if mock_target:
        http = _mock_target_client(config)
        url = "in-process mock target"
    else:
        url = target_url or config.target.url
        # Small local models can take a while per answer, so allow a long timeout.
        http = httpx.Client(base_url=url, timeout=300)

    judge = JudgePipeline(SYSTEM_PROMPT, config.judge.overlap_threshold)
    runs_dir = output_dir or resolve_path(config.output.runs_dir)
    total = len(attacks) * len(configs)
    console.print(
        f"Running [bold]{len(attacks)}[/] attacks x {len(configs)} defense configuration(s) "
        f"against [cyan]{url}[/]"
    )

    try:
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Attacking", total=total)
            summary = run_scan(
                attacks,
                TargetClient(http),
                judge,
                configs,
                runs_dir=runs_dir,
                target_url=url,
                on_result=lambda result: progress.advance(task),
            )
    except TargetError as exc:
        console.print(f"[red]{exc}[/]")
        console.print("Is the target running? Start it with: [bold]redrag serve-target[/]")
        raise typer.Exit(code=1) from exc
    finally:
        http.close()

    report_path = write_markdown_report(summary.run_dir)
    _print_asr_table(summary.results)
    console.print(f"Results: {summary.run_dir / 'results.jsonl'}")
    console.print(f"Report:  [bold]{report_path}[/]")


@app.command()
def report(
    run_dir: Path | None = typer.Argument(
        None, help="Run folder to report on. Default: the most recent run."
    ),
) -> None:
    """Rebuild the Markdown report for a saved scan."""
    from redrag.eval.report import load_results, write_markdown_report

    if run_dir is None:
        run_dir = _latest_run_dir(resolve_path(load_config().output.runs_dir))
        if run_dir is None:
            console.print("[red]No runs found. Run `redrag scan` first.[/]")
            raise typer.Exit(code=1)
    try:
        report_path = write_markdown_report(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    _print_asr_table(load_results(run_dir))
    console.print(f"Report: [bold]{report_path}[/]")


def _mock_target_client(config: Config):
    """An in-process target: MockClient + hashing embedder + in-memory Chroma."""
    from fastapi.testclient import TestClient

    from redrag.llm import MockClient
    from redrag.target.app import create_app
    from redrag.target.rag import ChromaStore, HashingEmbedder, RAGPipeline

    pipeline = RAGPipeline(
        store=ChromaStore(path=None, collection="mock_target"),
        embedder=HashingEmbedder(),
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
        top_k=config.rag.top_k,
    )
    app_ = create_app(config=config, llm=MockClient(model="mock-victim"), pipeline=pipeline)
    return TestClient(app_)


def _latest_run_dir(runs_dir: Path) -> Path | None:
    """Run ids are timestamps, so the alphabetically last folder is the newest."""
    if not runs_dir.exists():
        return None
    runs = sorted(p for p in runs_dir.iterdir() if (p / "results.jsonl").exists())
    return runs[-1] if runs else None


def _print_asr_table(results) -> None:
    """Show the per-category ASR for each defense configuration in the terminal."""
    from redrag.eval.metrics import asr_table, defense_configs
    from redrag.eval.report import CATEGORY_NAMES

    configs = defense_configs(results)
    table = Table(title="Attack success rate (lower is better for the defender)")
    table.add_column("Category")
    for config in configs:
        table.add_column(config, justify="right")
    for row, stats in asr_table(results).items():
        name = CATEGORY_NAMES.get(row, row).strip("*")
        style = "bold" if row == "overall" else None
        table.add_row(name, *(stats[config].format() for config in configs), style=style)
    console.print(table)


@app.command("serve-target")
def serve_target(
    host: str | None = typer.Option(None, help="Address to bind. Default: target.host in config."),
    port: int | None = typer.Option(None, help="Port to bind. Default: target.port in config."),
    mock: bool = typer.Option(
        False, "--mock", help="Use the offline MockClient instead of Ollama."
    ),
    offline_embeddings: bool = typer.Option(
        False,
        "--offline-embeddings",
        help="Use the hashing embedder instead of downloading bge-small.",
    ),
) -> None:
    """Start the deliberately vulnerable target RAG app (FastAPI)."""
    # Imported here so `redrag --help` stays fast (these pull in FastAPI and Chroma).
    import uvicorn

    from redrag.target.app import create_app

    config = load_config()
    if mock:
        config.llm.provider = "mock"
    if offline_embeddings:
        config.embeddings.provider = "hashing"

    host = host or config.target.host
    port = port or config.target.port
    model = "MockClient" if config.llm.provider == "mock" else config.ollama.victim_model
    console.print(f"[bold]Campus Handbook Assistant[/] using [cyan]{model}[/]")
    console.print(f"Building the knowledge base ({config.embeddings.provider} embeddings)...")
    target_app = create_app(config)
    console.print(f"API docs: [link]http://{host}:{port}/docs[/]")
    uvicorn.run(target_app, host=host, port=port)


@app.command()
def mcp() -> None:
    """Start the RedRAG MCP server over stdio."""
    _not_built_yet("mcp", "M6")


@app.command("show-config")
def show_config() -> None:
    """Print the effective configuration (YAML + .env + environment). Secrets are hidden."""
    config = load_config()
    # mode="json" turns SecretStr into "**********", so the API key is never shown.
    data = config.model_dump(mode="json")
    console.print_json(json.dumps(data))


if __name__ == "__main__":
    app()
