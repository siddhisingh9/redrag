"""Command-line interface: `redrag scan | report | serve-target | mcp`.

Run it either way:
    python -m redrag.cli --help
    redrag --help            (after `pip install -e .`)

Built with Typer (turns typed Python functions into commands) and Rich (pretty
terminal output). In M0 only `show-config` does real work; the other commands
are placeholders that say which milestone builds them.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from redrag.config import load_config

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
def scan() -> None:
    """Run the attack suite against the target RAG app and save results."""
    _not_built_yet("scan", "M2")


@app.command()
def report() -> None:
    """Build a Markdown/HTML report from a saved scan."""
    _not_built_yet("report", "M2")


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
