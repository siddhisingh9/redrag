"""FastAPI app for the deliberately vulnerable "Campus Handbook Assistant".

Endpoints:
- POST /chat   {message, session_id} -> {answer, retrieved_doc_ids, defenses, blocked_by}
- POST /ingest {doc_id, text}        -> adds a document (this is how poisoned docs get in)
- POST /reset                        -> restores the clean knowledge base, clears chat sessions
- GET  /health                       -> status, model name, document count

Defenses are chosen per request with the `X-RedRAG-Defenses` header ("none",
"all", or a comma-separated list). Without the header, the flags in config are
used. That lets one scan compare "no defenses" with "all defenses" without
restarting the app.

Start it with `redrag serve-target` (or `redrag serve-target --mock` to run
offline with the MockClient). API docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from redrag import tracing
from redrag.config import Config, load_config
from redrag.llm import LLMClient, LLMError, Message, build_local_client
from redrag.target import defenses
from redrag.target.rag import SYSTEM_PROMPT, RAGPipeline, build_messages, build_pipeline

logger = logging.getLogger(__name__)

# How many earlier messages (user + assistant) are kept per session. Keeping a
# few is what makes multi-turn attacks possible; capping it keeps prompts short
# for a small model.
MAX_HISTORY_MESSAGES = 10


# ----------------------------------------------------------- request/response


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    retrieved_doc_ids: list[str]
    # Extra fields (beyond the spec's {answer, retrieved_doc_ids}) that let the
    # scanner record exactly what happened.
    defenses: list[str] = Field(description="Defenses that were on for this request")
    blocked_by: str | None = Field(default=None, description="Defense that blocked, if any")


class IngestRequest(BaseModel):
    # Only simple ids, so a doc id can't be used to do anything odd later.
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    text: str = Field(min_length=1, max_length=20000)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class ResetResponse(BaseModel):
    status: str
    documents: int


class HealthResponse(BaseModel):
    status: str
    model: str
    documents: int
    chunks: int
    default_defenses: list[str]
    available_defenses: list[str]


# ---------------------------------------------------------------- the app


class Guard(Protocol):
    """What judge_guard needs: the fine-tuned classifier's goal-agnostic score."""

    def score_any_goal(self, answers: list[str]) -> float: ...


def create_app(
    config: Config | None = None,
    llm: LLMClient | None = None,
    pipeline: RAGPipeline | None = None,
    guard: Guard | None = None,
) -> FastAPI:
    """Build the target app.

    Everything can be passed in, so tests can use the MockClient, the offline
    hashing embedder and a temporary Chroma folder. With no arguments, it uses
    the real setup from config.

    `guard` is the fine-tuned judge used by the judge_guard defense. Without
    one, judge_guard is unavailable (and "all" means the other three defenses).
    `redrag serve-target` loads it automatically once a model has been trained.

    The knowledge base is rebuilt from the clean handbook on startup, so poisoned
    documents from a previous run never linger.
    """
    config = config or load_config()
    llm = llm or build_local_client(config, "victim")
    pipeline = pipeline or build_pipeline(config)
    pipeline.reset()
    available = list(defenses.AVAILABLE_DEFENSES) + (["judge_guard"] if guard else [])

    app = FastAPI(
        title="Campus Handbook Assistant (RedRAG target)",
        description="A deliberately vulnerable RAG chatbot for red-team testing. "
        "All secrets in it are fake canaries.",
        version="0.1.0",
    )

    # Chat history per session_id, kept in memory only.
    sessions: dict[str, list[Message]] = {}
    # FastAPI runs these handlers in a thread pool, so two requests can arrive
    # at once. The lock stops one request from reading the knowledge base or a
    # session while another is changing it.
    lock = threading.Lock()

    def active_defenses(header: str | None) -> frozenset[str]:
        try:
            return defenses.parse_defenses(header, config.defenses, available)
        except defenses.DefenseConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        with lock:
            documents = len(pipeline.doc_ids())
            chunks = pipeline.count()
        return HealthResponse(
            status="ok",
            model=llm.model,
            documents=documents,
            chunks=chunks,
            default_defenses=defenses.describe(active_defenses(None)),
            available_defenses=defenses.describe(available),
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        http_request: Request,
        x_redrag_defenses: str | None = Header(default=None, alias="X-RedRAG-Defenses"),
    ) -> ChatResponse:
        active = active_defenses(x_redrag_defenses)
        # If the scanner sent a trace id, our spans join its trace.
        with tracing.span(
            "target /chat",
            input=request.message,
            attributes={
                "redrag.session_id": request.session_id,
                "redrag.defenses": defenses.describe(active),
            },
            context=tracing.extract_context(http_request.headers),
        ) as chat_span:
            response = answer_question(request, active)
            tracing.set_output(chat_span, response.answer)
            tracing.set_attributes(chat_span, {"redrag.blocked_by": response.blocked_by})
            return response

    def answer_question(request: ChatRequest, active: frozenset[str]) -> ChatResponse:
        """The RAG pipeline with its defenses; each step is a span when tracing is on."""
        # Defense 1: block obvious injection phrases before doing anything else.
        if "input_filter" in active:
            with tracing.span("input_filter", kind="GUARDRAIL", input=request.message) as span:
                check = defenses.check_input(request.message)
                tracing.set_attributes(
                    span, {"output.value": "blocked" if check.blocked else "passed"}
                )
            if check.blocked:
                answer = defenses.INPUT_REFUSAL
                _remember(sessions, lock, request, answer)
                return ChatResponse(
                    answer=answer,
                    retrieved_doc_ids=[],
                    defenses=defenses.describe(active),
                    blocked_by="input_filter",
                )

        with tracing.span("retrieve", kind="RETRIEVER", input=request.message) as span:
            with lock:
                chunks = pipeline.retrieve(request.message)
                history = list(sessions.get(request.session_id, []))
            tracing.set_attributes(span, tracing.retrieval_attributes(chunks))

        # Defense 2 happens inside build_messages (sanitise=True).
        messages = build_messages(
            request.message,
            chunks,
            history=history,
            sanitise="context_sanitiser" in active,
        )

        with tracing.span(
            "llm", kind="LLM", attributes=tracing.llm_attributes(llm.model, messages)
        ) as span:
            try:
                answer = llm.generate(messages)
            except LLMError as exc:
                raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
            tracing.set_attributes(span, tracing.llm_attributes(llm.model, [], answer))
            tracing.set_output(span, answer)

        # Defense 3: check the answer before it leaves the app.
        blocked_by = None
        if "output_leak_filter" in active:
            with tracing.span("output_leak_filter", kind="GUARDRAIL", input=answer) as span:
                result = defenses.check_output(answer, SYSTEM_PROMPT, user_message=request.message)
                tracing.set_attributes(span, {"output.value": result.reason or "passed"})
            if result.blocked:
                logger.info("output_leak_filter blocked an answer: %s", result.reason)
                answer = defenses.OUTPUT_REFUSAL
                blocked_by = "output_leak_filter"

        # Defense 4: the fine-tuned judge reads the answer; block if it looks like
        # a successful attack. The goal is unknown here, so it checks all goal types.
        if "judge_guard" in active and blocked_by is None and guard is not None:
            with tracing.span("judge_guard", kind="GUARDRAIL", input=answer) as span:
                probability = guard.score_any_goal([answer])
                tracing.set_attributes(span, {"redrag.p_success": probability})
            if probability >= config.judge.guard_threshold:
                logger.info("judge_guard blocked an answer (P(success) = %.2f)", probability)
                answer = defenses.OUTPUT_REFUSAL
                blocked_by = "judge_guard"

        _remember(sessions, lock, request, answer)
        return ChatResponse(
            answer=answer,
            retrieved_doc_ids=_unique([chunk.doc_id for chunk in chunks]),
            defenses=defenses.describe(active),
            blocked_by=blocked_by,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        with lock:
            chunks = pipeline.ingest(request.doc_id, request.text)
        return IngestResponse(doc_id=request.doc_id, chunks=chunks)

    @app.post("/reset", response_model=ResetResponse)
    def reset() -> ResetResponse:
        with lock:
            documents = pipeline.reset()
            sessions.clear()
        return ResetResponse(status="ok", documents=documents)

    return app


def _remember(
    sessions: dict[str, list[Message]],
    lock: threading.Lock,
    request: ChatRequest,
    answer: str,
) -> None:
    """Add this turn to the session history, keeping only the latest messages."""
    with lock:
        history = sessions.setdefault(request.session_id, [])
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": answer})
        del history[:-MAX_HISTORY_MESSAGES]


def _unique(items: list[str]) -> list[str]:
    """Remove duplicates but keep the original order (best match first)."""
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
