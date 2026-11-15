"""Tracing with OpenTelemetry, viewed in Arize Phoenix.

What tracing gives you: a timeline of everything that happened for one attack:
the HTTP call to the target, what the retriever found (with scores), the exact
prompt and answer of the LLM, which defense blocked what, and how each judge
layer voted, with timings. When a result looks odd in the report, the trace
shows *why*.

Terms:
- a **span** is one step (e.g. "retrieve"), with a start, an end and attributes;
- a **trace** is a tree of spans for one unit of work. In RedRAG that's one
  attack, or one harmless question.

The scanner and the target app are separate processes. To join their spans
into one trace, the scanner puts the trace id in the HTTP request headers
(the W3C `traceparent` header) and the target app reads it back. That's called
*context propagation*.

Spans use OpenInference attribute names (`openinference.span.kind`,
`input.value`, `llm.model_name`, `retrieval.documents.N.document.content`, ...)
so Phoenix shows LLM calls and retrieved documents in its dedicated views.

Tracing is OFF unless enabled (`--trace`, or `tracing.enabled: true`). When off,
every helper here is a cheap no-op, so tests and normal scans don't need Phoenix.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import propagate, trace
from opentelemetry.context import Context

logger = logging.getLogger(__name__)

TRACER_NAME = "redrag"
MAX_ATTRIBUTE_CHARS = 4000  # keep huge prompts from bloating the trace store

_provider = None  # our TracerProvider when tracing is on, else None


def setup_tracing(
    endpoint: str = "http://localhost:6006/v1/traces",
    project: str = "redrag",
    service_name: str = "redrag-scanner",
    exporter: Any = None,
) -> None:
    """Start sending spans to Phoenix (or to `exporter`, which tests use).

    We keep our own provider instead of OpenTelemetry's global one, which can
    only be set once per process. That makes it possible for tests to switch
    tracing on and off.
    """
    global _provider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    shutdown_tracing()
    resource = Resource.create(
        {"service.name": service_name, "openinference.project.name": project}
    )
    provider = TracerProvider(resource=resource)
    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Batch: spans are sent in the background, so tracing doesn't slow the scan.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    _provider = provider
    logger.info("Tracing on: %s -> %s (project %s)", service_name, endpoint, project)


def shutdown_tracing() -> None:
    """Send any buffered spans and switch tracing off."""
    global _provider
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
        _provider = None


def flush() -> None:
    """Send buffered spans now (call at the end of a scan)."""
    if _provider is not None:
        _provider.force_flush()


def enabled() -> bool:
    return _provider is not None


def get_tracer() -> trace.Tracer:
    if _provider is not None:
        return _provider.get_tracer(TRACER_NAME)
    return trace.NoOpTracer()


def _clip(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= MAX_ATTRIBUTE_CHARS else text[:MAX_ATTRIBUTE_CHARS] + "...[cut]"


def set_attributes(current: trace.Span, attributes: Mapping[str, Any]) -> None:
    """Set several attributes, skipping None and clipping long strings."""
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float):
            current.set_attribute(key, value)
        elif isinstance(value, list | tuple):
            current.set_attribute(key, [_clip(item) for item in value])
        else:
            current.set_attribute(key, _clip(value))


@contextmanager
def span(
    name: str,
    kind: str = "CHAIN",
    input: Any = None,
    attributes: Mapping[str, Any] | None = None,
    context: Context | None = None,
) -> Iterator[trace.Span]:
    """Open a span (as the current one) for a block of code.

    `kind` is the OpenInference span kind: CHAIN, LLM, RETRIEVER, EVALUATOR,
    GUARDRAIL, TOOL, ... It decides how Phoenix displays the step.
    """
    with get_tracer().start_as_current_span(name, context=context) as current:
        set_attributes(current, {"openinference.span.kind": kind, "input.value": input})
        set_attributes(current, attributes or {})
        yield current


def set_output(current: trace.Span, value: Any) -> None:
    set_attributes(current, {"output.value": value})


def llm_attributes(model: str, messages: list[dict], answer: str | None = None) -> dict:
    """OpenInference attributes for an LLM call (Phoenix renders these as a chat)."""
    attributes: dict[str, Any] = {"llm.model_name": model}
    for index, message in enumerate(messages):
        attributes[f"llm.input_messages.{index}.message.role"] = message["role"]
        attributes[f"llm.input_messages.{index}.message.content"] = message["content"]
    if answer is not None:
        attributes["llm.output_messages.0.message.role"] = "assistant"
        attributes["llm.output_messages.0.message.content"] = answer
    return attributes


def retrieval_attributes(chunks: list) -> dict:
    """OpenInference attributes for retrieved documents (id, text, similarity score)."""
    attributes: dict[str, Any] = {}
    for index, chunk in enumerate(chunks):
        prefix = f"retrieval.documents.{index}.document"
        attributes[f"{prefix}.id"] = chunk.chunk_id
        attributes[f"{prefix}.content"] = chunk.text
        attributes[f"{prefix}.score"] = float(chunk.score)
    return attributes


def inject_headers() -> dict[str, str]:
    """HTTP headers carrying the current trace id (empty when tracing is off)."""
    headers: dict[str, str] = {}
    if enabled():
        propagate.inject(headers)
    return headers


def extract_context(headers: Mapping[str, str]) -> Context | None:
    """Read a trace id from incoming HTTP headers, to continue that trace."""
    if not enabled():
        return None
    return propagate.extract(dict(headers))
