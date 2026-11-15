"""Shared pytest setup.

Tests must run offline with no API keys (spec rule 0.9). The autouse fixture
below makes sure a developer's real setup can never leak into a test:
- every RedRAG environment variable is removed,
- the `.env` file is replaced by one that doesn't exist (so a real
  GEMINI_API_KEY is never read and no quota is ever spent),
- the judge model folder points at an empty temp dir (so a trained model in
  models/ doesn't change test results), and labels/runs go to temp folders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from redrag.config import ENV_OVERRIDES


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in [*ENV_OVERRIDES, "REDRAG_CONFIG"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("redrag.config.DEFAULT_DOTENV_PATH", tmp_path / "no.env")
    monkeypatch.setenv("REDRAG_JUDGE_MODEL_DIR", str(tmp_path / "no-model"))
    monkeypatch.setenv("REDRAG_LABELS_FILE", str(tmp_path / "labels" / "hand_labels.jsonl"))
    monkeypatch.setenv("REDRAG_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("REDRAG_RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture
def pipeline(tmp_path: Path):
    """An offline RAG pipeline: hashing embedder + Chroma in a temp folder, clean handbook."""
    from redrag.target.rag import ChromaStore, HashingEmbedder, RAGPipeline

    rag = RAGPipeline(
        store=ChromaStore(tmp_path / "chroma", collection="test"),
        embedder=HashingEmbedder(),
    )
    rag.reset()
    return rag


@pytest.fixture(scope="session")
def tiny_judge_model(tmp_path_factory) -> Path:
    """A tiny, randomly initialised DeBERTa classifier saved like the notebook saves it.

    It's useless as a judge (random weights) but lets us test the real
    loading and inference code offline in a second, with no download.
    A word-piece tokenizer stands in for DeBERTa's (which needs a download).
    """
    import json

    from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2ForSequenceClassification

    from redrag.judge.classifier import METADATA_FILE

    model_dir = tmp_path_factory.mktemp("tiny-judge")
    words = [
        "the",
        "a",
        "assistant",
        "attacker",
        "code",
        "word",
        "secret",
        "prompt",
        "pineapple",
        "7",
        "canary",
        "i",
        "can't",
        "share",
    ]
    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=2,
    )
    model = DebertaV2ForSequenceClassification(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / METADATA_FILE).write_text(
        json.dumps({"name": "tiny-test-judge", "max_length": 64, "success_label_id": 1}),
        encoding="utf-8",
    )
    return model_dir
