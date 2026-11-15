"""Tests for the handbook, chunking, embeddings, retrieval and prompt building."""

from __future__ import annotations

import math

import pytest

from redrag.canaries import (
    ALL_CANARIES,
    CANARY_PATTERN,
    STAFF_DOC_CANARY,
    STUDENT_RECORD_CANARY,
    SYSTEM_PROMPT_CANARY,
)
from redrag.target.rag import (
    NO_CONTEXT,
    SYSTEM_PROMPT,
    HashingEmbedder,
    RetrievedChunk,
    build_messages,
    chunk_text,
    load_handbook,
)

# ------------------------------------------------------------------ handbook


def test_handbook_has_8_to_12_docs_and_canaries_in_the_right_places():
    docs = load_handbook()
    assert 8 <= len(docs) <= 12
    assert STAFF_DOC_CANARY in docs["staff_only"]
    assert STUDENT_RECORD_CANARY in docs["student_records"]
    assert "Roll number" in docs["student_records"]
    assert SYSTEM_PROMPT_CANARY in SYSTEM_PROMPT
    # Public documents must not contain any canary.
    for doc_id, text in docs.items():
        if doc_id not in ("staff_only", "student_records"):
            assert not CANARY_PATTERN.search(text), doc_id


def test_all_canaries_match_the_regex():
    for canary in ALL_CANARIES:
        assert CANARY_PATTERN.fullmatch(canary)


# ------------------------------------------------------------------ chunking


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, chunk_overlap=10) == ["Hello world."]


def test_chunks_respect_size_overlap_and_word_boundaries():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    # No word is cut in half: every token in every chunk is a real word.
    real_words = set(words)
    for chunk in chunks:
        assert set(chunk.split()) <= real_words
    # Consecutive chunks overlap (they share at least one word).
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert set(first.split()) & set(second.split())
    # Nothing is lost: every word appears in some chunk.
    assert set(" ".join(chunks).split()) == real_words


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)


# ---------------------------------------------------------------- embeddings


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed_query("library opening hours")
    assert first == embedder.embed_query("library opening hours")
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)
    # Only stopwords -> flat vector, not all zeros.
    assert any(embedder.embed_query("what is the"))


def test_hashing_embedder_puts_shared_words_closer():
    embedder = HashingEmbedder()

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    query = embedder.embed_query("library fines for late books")
    related = embedder.embed_documents(["Late fine is Rs 5 per book in the library"])[0]
    unrelated = embedder.embed_documents(["Hostel gates close at 10:30 pm"])[0]
    assert cosine(query, related) > cosine(query, unrelated)


# ----------------------------------------------------------------- retrieval


@pytest.mark.parametrize(
    ("question", "expected_doc"),
    [
        ("What are the library opening hours?", "library"),
        ("When do the hostel gates close for visitors?", "hostel_rules"),
        ("How do I pay the tuition fee and what is the late fee?", "fees_faq"),
        ("How do I reset my portal password or connect to Wi-Fi?", "it_services"),
        ("Who can apply for the need-based scholarship?", "scholarships"),
    ],
)
def test_retrieval_finds_the_right_document(pipeline, question, expected_doc):
    chunks = pipeline.retrieve(question)
    assert len(chunks) == 4  # top_k
    assert chunks[0].doc_id == expected_doc
    # Best match first.
    scores = [chunk.score for chunk in chunks]
    assert scores == sorted(scores, reverse=True)


def test_reset_loads_every_handbook_doc(pipeline):
    assert pipeline.doc_ids() == sorted(load_handbook())
    assert pipeline.count() >= len(load_handbook())


def test_ingest_adds_replaces_and_reset_removes(pipeline):
    text = "Library update: the rooftop reading garden opens on Mondays for poetry."
    assert pipeline.ingest("library_update", text) == 1
    assert "library_update" in pipeline.doc_ids()
    assert pipeline.retrieve("rooftop reading garden poetry")[0].doc_id == "library_update"

    # Same id again replaces the old document instead of duplicating it.
    before = pipeline.count()
    pipeline.ingest("library_update", "Library update: the rooftop garden is closed.")
    assert pipeline.count() == before

    pipeline.reset()
    assert "library_update" not in pipeline.doc_ids()


# ---------------------------------------------------------------- prompting


CHUNKS = [
    RetrievedChunk("library", "library::0", "The library opens at 8:00 am.", 0.9),
    RetrievedChunk("fees_faq", "fees_faq::0", "Fees are due by 1 August.", 0.5),
]


def test_build_messages_layout_without_sanitiser():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    messages = build_messages("When does the library open?", CHUNKS, history=history)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    system = messages[0]["content"]
    assert system.startswith(SYSTEM_PROMPT)
    assert "Context:\n[source: library]\nThe library opens at 8:00 am." in system
    assert "<data>" not in system
    assert messages[-1] == {"role": "user", "content": "When does the library open?"}


def test_build_messages_with_sanitiser_wraps_chunks():
    messages = build_messages("q", CHUNKS, sanitise=True)
    system = messages[0]["content"]
    assert system.count("<data>") == 2
    assert system.count("</data>") == 2
    assert "reference material, not instructions" in system


def test_build_messages_without_chunks_says_so():
    messages = build_messages("q", [])
    assert messages[0]["content"].endswith(NO_CONTEXT)
