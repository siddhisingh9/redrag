"""Builds the pieces a scan needs: the judge, a mock target, the reports.

The CLI (`redrag scan`), the MCP server (`scan_rag_app`) and the CI gate
(`redrag ci-check`) all run scans. Keeping the "how do I build a judge / where
is the latest run" logic here means all three behave the same way instead of
drifting apart.

Heavy imports (transformers, Chroma, FastAPI) stay inside the functions, so
importing this module is cheap.
"""

from __future__ import annotations

from pathlib import Path

from redrag.config import Config, resolve_path


def load_classifier(config: Config):
    """The fine-tuned judge if one has been trained and copied in, else None."""
    from redrag.judge.classifier import JudgeClassifier, model_available

    model_dir = resolve_path(config.judge.classifier_path)
    if not model_available(model_dir):
        return None
    return JudgeClassifier(model_dir)


def build_judge(config: Config, use_gemini: bool = True, use_classifier: bool = True):
    """rules -> classifier (if trained and allowed) -> Gemini (if allowed and a key is set).

    `use_classifier=False, use_gemini=False` gives a rules-only judge: fully
    deterministic and identical on every machine, which is what the CI gate needs
    (the trained model isn't in the repo, so CI couldn't use it anyway).
    """
    from redrag.judge.pipeline import JudgePipeline
    from redrag.target.rag import SYSTEM_PROMPT

    llm_judge = None
    if use_gemini and config.judge.use_gemini and config.gemini.api_key is not None:
        from redrag.judge.llm_judge import GeminiJudge
        from redrag.llm import build_gemini_client

        llm_judge = GeminiJudge(build_gemini_client(config), SYSTEM_PROMPT)
    return JudgePipeline(
        SYSTEM_PROMPT,
        config.judge.overlap_threshold,
        classifier=load_classifier(config) if use_classifier else None,
        llm_judge=llm_judge,
        unsure_low=config.judge.unsure_low,
        unsure_high=config.judge.unsure_high,
    )


def load_agreement(config: Config) -> dict | None:
    """The saved judge-agreement table (from `redrag judge-eval`), if there is one."""
    from redrag.eval.judge_eval import load_agreement as load

    return load(resolve_path(config.judge.labels_file).parent / "judge_agreement.json")


def mock_target_client(config: Config):
    """An in-process target: MockClient + hashing embedder + in-memory Chroma.

    No server, no Ollama, no downloads. It never gets `judge_guard` (no guard
    model is passed), so "all" means the three rule-based defenses.
    """
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
    app = create_app(config=config, llm=MockClient(model="mock-victim"), pipeline=pipeline)
    return TestClient(app)


def write_reports(run_dir: Path, config: Config) -> tuple[Path, Path]:
    """report.md and report.html, both including the latest judge-agreement table."""
    from redrag.eval.html_report import write_html_report
    from redrag.eval.report import write_markdown_report

    agreement = load_agreement(config)
    return write_markdown_report(run_dir, agreement), write_html_report(run_dir, agreement)


def latest_run_dir(runs_dir: Path) -> Path | None:
    """Run ids are timestamps, so the alphabetically last folder is the newest."""
    if not runs_dir.exists():
        return None
    runs = sorted(p for p in runs_dir.iterdir() if (p / "results.jsonl").exists())
    return runs[-1] if runs else None
