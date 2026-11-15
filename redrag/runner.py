"""Orchestrates a scan: send every attack to the target, judge it, save the result.

For each defense configuration (e.g. "none", then "all"), and each attack:
1. If the knowledge base may be dirty (a poisoned doc was added), POST /reset.
2. If the attack has a poison document, POST /ingest it.
3. Send each turn to POST /chat in a fresh session, with the
   X-RedRAG-Defenses header, timing every call.
4. Judge the answers and append an AttackResult line to results.jsonl.

Results are written as they come in, so a scan stopped halfway still leaves
usable data behind.

The building blocks (`start_run`, `AttackExecutor`, `RunRecorder`) are shared
with the adaptive attacker agent (redrag/agent/attacker_graph.py), which runs
attacks one at a time and decides what to try next.

The target is reached through any `httpx.Client`. For a real server that's
`httpx.Client(base_url=...)`. In tests and `--mock-target` mode it's FastAPI's
`TestClient`, which is also an httpx.Client but calls the app in-process.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import httpx
from opentelemetry import trace

from redrag import tracing
from redrag.attacks.schema import Attack, AttackResult, Turn
from redrag.eval.benign import BENIGN_RESULTS_FILE, BenignQuestion, BenignResult, is_useful
from redrag.eval.report import RESULTS_FILE, RUN_INFO_FILE
from redrag.judge.pipeline import JudgePipeline
from redrag.target.defenses import DEFENSE_NAMES


class TargetError(RuntimeError):
    """The target app could not be reached or returned an error."""


class TargetClient:
    """Small wrapper around the target's HTTP API."""

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def health(self) -> dict:
        return self._request("GET", "/health")

    def reset(self) -> dict:
        return self._request("POST", "/reset")

    def ingest(self, doc_id: str, text: str) -> dict:
        return self._request("POST", "/ingest", json={"doc_id": doc_id, "text": text})

    def chat(self, message: str, session_id: str, defenses: str) -> dict:
        return self._request(
            "POST",
            "/chat",
            json={"message": message, "session_id": session_id},
            headers={"X-RedRAG-Defenses": defenses},
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        # Carry the current trace id, so the target's spans join this trace.
        headers = {**tracing.inject_headers(), **kwargs.pop("headers", {})}
        try:
            response = self.http.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise TargetError(f"Could not reach the target ({method} {path}): {exc}") from exc
        if response.status_code != 200:
            raise TargetError(
                f"{method} {path} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()


def normalise_defense_config(value: str) -> str:
    """Turn user input like "Output_Leak_Filter, input_filter" into a stable label.

    "none" and "all" stay as they are; a list is lowercased, de-duplicated and
    put in a fixed order, so the same configuration always gets the same label.
    """
    value = value.strip().lower()
    if value in ("", "none"):
        return "none"
    if value == "all":
        return "all"
    names = {part.strip() for part in value.split(",") if part.strip()}
    unknown = names - set(DEFENSE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown or unavailable defense(s): {sorted(unknown)}. "
            f"Choose from {list(DEFENSE_NAMES)}, or 'none' / 'all'."
        )
    return ",".join(name for name in DEFENSE_NAMES if name in names)


def new_run_id() -> str:
    """A timestamp id like 20261115-153000 (no colons, so it's a valid Windows folder name)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def defense_combinations(available: list[str]) -> list[str]:
    """Every on/off combination of the available defenses, as config labels.

    With 3 defenses that's 2^3 = 8 configurations; with judge_guard, 16. Ordered
    from fewest to most defenses, so reports read naturally from "none" to "all":
        none, input_filter, context_sanitiser, ..., input_filter,context_sanitiser, ..., all
    """
    from itertools import combinations

    ordered = [name for name in DEFENSE_NAMES if name in available]
    labels = []
    for size in range(len(ordered) + 1):
        for combo in combinations(ordered, size):
            if size == 0:
                labels.append("none")
            elif size == len(ordered):
                labels.append("all")
            else:
                labels.append(",".join(combo))
    return labels


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    target_model: str


def start_run(
    target: TargetClient,
    judge: JudgePipeline,
    defense_configs: list[str],
    attacks: list[Attack],
    runs_dir: Path,
    target_url: str,
    run_id: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> RunContext:
    """Check the target is up, create the run folder and write run.json."""
    run_id = run_id or new_run_id()
    run_dir = runs_dir / run_id
    health = target.health()  # fails fast if the target isn't running
    run_dir.mkdir(parents=True, exist_ok=False)
    target_model = health.get("model", "unknown")

    info = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "scan",
        "target_url": target_url,
        "target_model": target_model,
        "available_defenses": health.get("available_defenses", []),
        "defense_configs": defense_configs,
        "attack_count": len(attacks),
        "attack_ids": [attack.id for attack in attacks],
        "judge_layers": judge.layers,
        **(extra_info or {}),
    }
    (run_dir / RUN_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return RunContext(run_id=run_id, run_dir=run_dir, target_model=target_model)


class RunRecorder:
    """Appends each result to a JSONL file as soon as it exists, and keeps a copy.

    Used for attack results (results.jsonl) and benign results (benign.jsonl).
    """

    def __init__(
        self,
        run_dir: Path,
        on_result: Callable[[Any], None] | None = None,
        filename: str = RESULTS_FILE,
    ) -> None:
        self.results: list[Any] = []
        self._on_result = on_result
        self._file: IO[str] = (run_dir / filename).open("w", encoding="utf-8")

    def record(self, result: Any) -> None:
        self._file.write(result.model_dump_json() + "\n")
        self._file.flush()  # so a stopped scan still has every finished result
        self.results.append(result)
        if self._on_result is not None:
            self._on_result(result)

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()


@dataclass
class Exchange:
    """The raw conversation with the target for one attack, before judging."""

    transcript: list[Turn] = field(default_factory=list)
    defenses_on: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    # Tracing: the attack's root span stays open from send() until it is judged.
    trace_span: Any = None
    trace_context: Any = None

    def retrieved_doc_ids(self) -> list[str]:
        """Every doc retrieved in any turn, without duplicates, in first-seen order."""
        seen: list[str] = []
        for turn in self.transcript:
            for doc_id in turn.retrieved_doc_ids:
                if doc_id not in seen:
                    seen.append(doc_id)
        return seen

    def first_blocked_by(self) -> str | None:
        for turn in self.transcript:
            if turn.blocked_by:
                return turn.blocked_by
        return None


class AttackExecutor:
    """Runs single attacks against the target and keeps the knowledge base clean.

    It remembers whether the last attack added a poisoned document, and only
    resets the target (which re-embeds the whole handbook) when that happened.
    """

    def __init__(
        self, target: TargetClient, judge: JudgePipeline, run_id: str, target_model: str
    ) -> None:
        self.target = target
        self.judge = judge
        self.run_id = run_id
        self.target_model = target_model
        self._dirty = True  # we don't know what state the target is in yet

    def reset(self) -> None:
        self.target.reset()
        self._dirty = False

    def ask_benign(self, question: BenignQuestion, defense_config: str) -> BenignResult:
        """Ask one harmless question (on a clean knowledge base) and record what happened."""
        if self._dirty:
            self.reset()  # a leftover poisoned doc would make the FRR unfair
        session_id = f"{self.run_id}:{defense_config}:{question.id}"
        started = time.perf_counter()
        answer, blocked_by, retrieved, defenses_on, error = "", None, [], [], None
        with tracing.span(
            f"benign {question.id} [{defense_config}]",
            input=question.question,
            attributes={"redrag.defense_config": defense_config, "redrag.tricky": question.tricky},
        ) as benign_span:
            try:
                response = self.target.chat(question.question, session_id, defense_config)
                answer = response["answer"]
                blocked_by = response.get("blocked_by")
                retrieved = response.get("retrieved_doc_ids", [])
                defenses_on = response.get("defenses", [])
            except TargetError as exc:
                error = str(exc)
            tracing.set_output(benign_span, answer)
            tracing.set_attributes(benign_span, {"redrag.blocked_by": blocked_by})
        return BenignResult(
            run_id=self.run_id,
            question_id=question.id,
            topic=question.topic,
            tricky=question.tricky,
            answerable=question.answerable,
            defense_config=defense_config,
            defenses=defenses_on,
            question=question.question,
            answer=answer,
            retrieved_doc_ids=retrieved,
            blocked_by=blocked_by,
            useful=error is None and is_useful(question, answer, blocked_by),
            latency_ms=(time.perf_counter() - started) * 1000,
            timestamp=datetime.now(timezone.utc),
            error=error,
        )

    def execute(self, attack: Attack, defense_config: str, attempt: int = 0) -> AttackResult:
        """Send one attack (all its turns) and judge it. Target errors are recorded, not raised."""
        return self.judge_exchange(
            attack, defense_config, self.send(attack, defense_config), attempt
        )

    def send(self, attack: Attack, defense_config: str) -> Exchange:
        """Talk to the target: reset if needed, ingest any poison, send every turn."""
        if self._dirty:
            self.reset()

        # A fresh session per attack (and per configuration), so earlier attacks
        # can't influence this one through chat history.
        session_id = f"{self.run_id}:{defense_config}:{attack.id}"
        exchange = Exchange()
        started = time.perf_counter()
        # One trace per attack. The root span is opened here and closed in
        # judge_exchange(), because the agent sends and judges in separate steps.
        root = tracing.get_tracer().start_span(f"attack {attack.id} [{defense_config}]")
        tracing.set_attributes(
            root,
            {
                "openinference.span.kind": "CHAIN",
                "input.value": " / ".join(attack.turns),
                "redrag.attack_id": attack.id,
                "redrag.seed_id": attack.root_id,
                "redrag.category": attack.category,
                "redrag.defense_config": defense_config,
                "redrag.strategy": attack.strategy,
                "redrag.run_id": self.run_id,
            },
        )
        exchange.trace_span = root
        exchange.trace_context = trace.set_span_in_context(root)

        try:
            if attack.poison_doc is not None:
                self._dirty = True  # set first, in case ingest half-succeeds
                with tracing.span(
                    "target.ingest",
                    input=attack.poison_doc.text,
                    attributes={"redrag.doc_id": attack.poison_doc.doc_id},
                    context=exchange.trace_context,
                ):
                    self.target.ingest(attack.poison_doc.doc_id, attack.poison_doc.text)
            for number, message in enumerate(attack.turns, start=1):
                turn_started = time.perf_counter()
                with tracing.span(
                    f"target.chat turn {number}", input=message, context=exchange.trace_context
                ) as chat_span:
                    response = self.target.chat(message, session_id, defense_config)
                    tracing.set_output(chat_span, response["answer"])
                    tracing.set_attributes(
                        chat_span, {"redrag.blocked_by": response.get("blocked_by")}
                    )
                exchange.transcript.append(
                    Turn(
                        user=message,
                        answer=response["answer"],
                        retrieved_doc_ids=response.get("retrieved_doc_ids", []),
                        blocked_by=response.get("blocked_by"),
                        latency_ms=(time.perf_counter() - turn_started) * 1000,
                    )
                )
                exchange.defenses_on = response.get("defenses", exchange.defenses_on)
        except TargetError as exc:
            exchange.error = str(exc)

        exchange.latency_ms = (time.perf_counter() - started) * 1000
        return exchange

    def judge_exchange(
        self, attack: Attack, defense_config: str, exchange: Exchange, attempt: int = 0
    ) -> AttackResult:
        """Judge what came back and package everything as an AttackResult."""
        transcript = exchange.transcript
        if exchange.error is None:
            with tracing.span(
                "judge", kind="EVALUATOR", context=exchange.trace_context
            ) as judge_span:
                decision = self.judge.judge(attack, transcript)
                tracing.set_output(judge_span, decision.final_verdict)
            verdicts = decision.verdicts
            final, decided_by = decision.final_verdict, decision.decided_by
        else:
            verdicts, final, decided_by = [], "error", "none"

        root = exchange.trace_span
        if root is not None:
            tracing.set_attributes(
                root,
                {"redrag.verdict": final, "redrag.decided_by": decided_by, "output.value": final},
            )
            if exchange.error is not None:
                root.set_status(trace.Status(trace.StatusCode.ERROR, exchange.error))
            root.end()

        return AttackResult(
            run_id=self.run_id,
            attack_id=attack.id,
            parent_id=attack.parent_id,
            seed_id=attack.root_id,
            attempt=attempt,
            strategy=attack.strategy,
            notes=attack.notes,
            success_signal=attack.success_signal,
            category=attack.category,
            owasp=attack.owasp,
            language=attack.language,
            defense_config=defense_config,
            defenses=exchange.defenses_on,
            poison_doc_id=attack.poison_doc.doc_id if attack.poison_doc else None,
            poison_doc_text=attack.poison_doc.text if attack.poison_doc else None,
            transcript=transcript,
            retrieved_doc_ids=exchange.retrieved_doc_ids(),
            judge_verdicts=verdicts,
            final_verdict=final,
            decided_by=decided_by,
            latency_ms=exchange.latency_ms,
            model_names={"target": self.target_model, "judge": "+".join(self.judge.layers)},
            timestamp=datetime.now(timezone.utc),
            error=exchange.error,
        )


# --------------------------------------------------------------------------
# The plain scan
# --------------------------------------------------------------------------


@dataclass
class ScanSummary:
    run_id: str
    run_dir: Path
    results: list[AttackResult] = field(default_factory=list)
    benign_results: list[BenignResult] = field(default_factory=list)


def run_scan(
    attacks: list[Attack],
    target: TargetClient,
    judge: JudgePipeline,
    defense_configs: list[str],
    runs_dir: Path,
    target_url: str,
    run_id: str | None = None,
    on_result: Callable[[AttackResult], None] | None = None,
    benign_questions: list[BenignQuestion] | None = None,
    on_benign: Callable[[BenignResult], None] | None = None,
) -> ScanSummary:
    """Run every attack once against every defense configuration and save the results.

    Args:
        attacks: The attacks to run.
        target: Client for the target app.
        judge: Judge pipeline that decides success/fail.
        defense_configs: e.g. ["none", "all"]; see `normalise_defense_config`.
        runs_dir: Parent folder; results go in `runs_dir/<run_id>/`.
        target_url: Recorded in run.json for the report.
        run_id: Optional fixed id (tests); defaults to a timestamp.
        on_result: Called after each attack, e.g. to update a progress bar.
        benign_questions: Harmless questions to ask under every configuration
            too (for the false-refusal rate). Saved to benign.jsonl.
        on_benign: Called after each harmless question.
    """
    configs = [normalise_defense_config(config) for config in defense_configs]
    context = start_run(
        target,
        judge,
        configs,
        attacks,
        runs_dir,
        target_url,
        run_id,
        extra_info={"benign_count": len(benign_questions or [])},
    )
    executor = AttackExecutor(target, judge, context.run_id, context.target_model)

    with (
        RunRecorder(context.run_dir, on_result) as recorder,
        RunRecorder(context.run_dir, on_benign, BENIGN_RESULTS_FILE) as benign_recorder,
    ):
        for config in configs:
            for attack in attacks:
                recorder.record(executor.execute(attack, config))
            run_benign(executor, benign_questions or [], config, benign_recorder.record)
        executor.reset()  # leave the target clean for whoever uses it next

    return ScanSummary(
        run_id=context.run_id,
        run_dir=context.run_dir,
        results=recorder.results,
        benign_results=benign_recorder.results,
    )


def run_benign(
    executor: AttackExecutor,
    questions: list[BenignQuestion],
    defense_config: str,
    record: Callable[[BenignResult], None],
) -> None:
    """Ask every harmless question under one defense configuration."""
    for question in questions:
        record(executor.ask_benign(question, defense_config))


# --------------------------------------------------------------------------
# Re-judging saved results
# --------------------------------------------------------------------------


def rejudge(results: list[AttackResult], judge: JudgePipeline) -> list[AttackResult]:
    """Run a (newer) judge pipeline on saved transcripts, without contacting the target.

    Useful after training the classifier: old scans get the improved verdicts
    for free. Errors stay errors (there is nothing to judge).
    """
    from redrag.eval.labels import attack_from_result, seeds_by_id

    seeds = seeds_by_id()
    updated = []
    for result in results:
        if result.final_verdict == "error" or not result.transcript:
            updated.append(result)
            continue
        attack = attack_from_result(result, seeds)
        decision = judge.judge(attack, result.transcript)
        updated.append(
            result.model_copy(
                update={
                    "success_signal": attack.success_signal,
                    "judge_verdicts": decision.verdicts,
                    "final_verdict": decision.final_verdict,
                    "decided_by": decision.decided_by,
                    "model_names": {**result.model_names, "judge": "+".join(judge.layers)},
                }
            )
        )
    return updated
