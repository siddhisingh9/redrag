"""Orchestrates a scan: send every attack to the target, judge it, save the result.

For each defense configuration (e.g. "none", then "all"), and each attack:
1. If the knowledge base may be dirty (a poisoned doc was added), POST /reset.
2. If the attack has a poison document, POST /ingest it.
3. Send each turn to POST /chat in a fresh session, with the
   X-RedRAG-Defenses header, timing every call.
4. Judge the answers and append an AttackResult line to results.jsonl.

Results are written as they come in, so a scan stopped halfway still leaves
usable data behind.

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

import httpx

from redrag.attacks.schema import Attack, AttackResult, Turn
from redrag.eval.report import RESULTS_FILE, RUN_INFO_FILE
from redrag.judge.pipeline import JudgePipeline
from redrag.target.defenses import AVAILABLE_DEFENSES


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
        try:
            response = self.http.request(method, path, **kwargs)
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
    unknown = names - set(AVAILABLE_DEFENSES)
    if unknown:
        raise ValueError(
            f"Unknown or unavailable defense(s): {sorted(unknown)}. "
            f"Choose from {list(AVAILABLE_DEFENSES)}, or 'none' / 'all'."
        )
    return ",".join(name for name in AVAILABLE_DEFENSES if name in names)


def new_run_id() -> str:
    """A timestamp id like 20261115-153000 (no colons, so it's a valid Windows folder name)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class ScanSummary:
    run_id: str
    run_dir: Path
    results: list[AttackResult] = field(default_factory=list)


def run_scan(
    attacks: list[Attack],
    target: TargetClient,
    judge: JudgePipeline,
    defense_configs: list[str],
    runs_dir: Path,
    target_url: str,
    run_id: str | None = None,
    on_result: Callable[[AttackResult], None] | None = None,
) -> ScanSummary:
    """Run every attack against every defense configuration and save the results.

    Args:
        attacks: The attacks to run.
        target: Client for the target app.
        judge: Judge pipeline that decides success/fail.
        defense_configs: e.g. ["none", "all"]; see `normalise_defense_config`.
        runs_dir: Parent folder; results go in `runs_dir/<run_id>/`.
        target_url: Recorded in run.json for the report.
        run_id: Optional fixed id (tests); defaults to a timestamp.
        on_result: Called after each attack, e.g. to update a progress bar.
    """
    configs = [normalise_defense_config(config) for config in defense_configs]
    run_id = run_id or new_run_id()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    health = target.health()  # fails fast if the target isn't running
    target_model = health.get("model", "unknown")

    info = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_url": target_url,
        "target_model": target_model,
        "defense_configs": configs,
        "attack_count": len(attacks),
        "attack_ids": [attack.id for attack in attacks],
        "judge_layers": judge.layers,
    }
    (run_dir / RUN_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

    summary = ScanSummary(run_id=run_id, run_dir=run_dir)
    results_path = run_dir / RESULTS_FILE
    with results_path.open("w", encoding="utf-8") as results_file:
        for config in configs:
            # Always start each configuration from a clean knowledge base.
            target.reset()
            dirty = False
            for attack in attacks:
                if dirty:
                    target.reset()
                    dirty = False
                result = run_attack(attack, target, judge, config, run_id, target_model)
                dirty = attack.poison_doc is not None

                results_file.write(result.model_dump_json() + "\n")
                results_file.flush()
                summary.results.append(result)
                if on_result is not None:
                    on_result(result)
        # Leave the target clean for whoever uses it next.
        target.reset()
    return summary


def run_attack(
    attack: Attack,
    target: TargetClient,
    judge: JudgePipeline,
    defense_config: str,
    run_id: str,
    target_model: str,
) -> AttackResult:
    """Run one attack (all its turns) and judge it. Errors are recorded, not raised."""
    # A fresh session per attack (and per configuration), so earlier attacks
    # can't influence this one through chat history.
    session_id = f"{run_id}:{defense_config}:{attack.id}"
    transcript: list[Turn] = []
    defenses_on: list[str] = []
    error: str | None = None
    started = time.perf_counter()

    try:
        if attack.poison_doc is not None:
            target.ingest(attack.poison_doc.doc_id, attack.poison_doc.text)
        for message in attack.turns:
            turn_started = time.perf_counter()
            response = target.chat(message, session_id, defense_config)
            transcript.append(
                Turn(
                    user=message,
                    answer=response["answer"],
                    retrieved_doc_ids=response.get("retrieved_doc_ids", []),
                    blocked_by=response.get("blocked_by"),
                    latency_ms=(time.perf_counter() - turn_started) * 1000,
                )
            )
            defenses_on = response.get("defenses", defenses_on)
    except TargetError as exc:
        error = str(exc)

    if error is None:
        decision = judge.judge(attack, transcript)
        verdicts, final, decided_by = decision.verdicts, decision.final_verdict, decision.decided_by
    else:
        verdicts, final, decided_by = [], "error", "none"

    retrieved: list[str] = []
    for turn in transcript:
        for doc_id in turn.retrieved_doc_ids:
            if doc_id not in retrieved:
                retrieved.append(doc_id)

    return AttackResult(
        run_id=run_id,
        attack_id=attack.id,
        parent_id=attack.parent_id,
        category=attack.category,
        owasp=attack.owasp,
        language=attack.language,
        defense_config=defense_config,
        defenses=defenses_on,
        poison_doc_id=attack.poison_doc.doc_id if attack.poison_doc else None,
        transcript=transcript,
        retrieved_doc_ids=retrieved,
        judge_verdicts=verdicts,
        final_verdict=final,
        decided_by=decided_by,
        latency_ms=(time.perf_counter() - started) * 1000,
        model_names={"target": target_model, "judge": "+".join(judge.layers)},
        timestamp=datetime.now(timezone.utc),
        error=error,
    )
