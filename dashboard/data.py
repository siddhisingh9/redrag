"""Data for the dashboard: loading runs and shaping them into tables.

No Streamlit in here, on purpose. The pages only draw; everything they show is
computed by plain functions that the tests can call directly. The numbers come
from the same `redrag.eval` functions as the Markdown/HTML reports, so the
dashboard and the reports always agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from redrag.attacks.schema import AttackResult
from redrag.eval.benign import BenignResult, load_benign_results
from redrag.eval.charts import short_label
from redrag.eval.metrics import best_per_seed, compute_asr, defense_configs, original_attempts
from redrag.eval.report import (
    CATEGORY_NAMES,
    RUN_INFO_FILE,
    DefenseRow,
    defense_summary,
    load_results,
)

CATEGORIES = [name for name in CATEGORY_NAMES if name != "overall"]


def category_name(category: str) -> str:
    return CATEGORY_NAMES.get(category, category).strip("*")


# ---------------------------------------------------------------------- runs


@dataclass
class RunInfo:
    """What the run picker needs to know about one saved scan, without loading results."""

    run_id: str
    path: Path
    mode: str  # "scan" or "agent"
    target_model: str
    configs: list[str]
    attack_count: int
    has_benign: bool
    title: str = ""  # set for the bundled demo scans (demo/runs/index.json)

    @property
    def label(self) -> str:
        """e.g. "20261116-034159 - scan - llama3.2:3b - 16 configurations"."""
        if self.title:
            return self.title
        configs = len(self.configs)
        noun = "configuration" if configs == 1 else "configurations"
        return f"{self.run_id} - {self.mode} - {self.target_model} - {configs} {noun}"


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Every saved scan in `runs_dir`, newest first (run ids are timestamps).

    If the folder has an `index.json` (the bundled demo scans do), its order and
    titles are used instead: `[{"run_id": ..., "title": ...}, ...]`.
    """
    if not runs_dir.exists():
        return []
    runs = _scan_folder(runs_dir)
    index_path = runs_dir / "index.json"
    if index_path.exists():
        by_id = {run.run_id: run for run in runs}
        ordered = []
        for entry in json.loads(index_path.read_text(encoding="utf-8")):
            run = by_id.pop(entry["run_id"], None)
            if run is not None:
                run.title = entry.get("title", "")
                ordered.append(run)
        runs = ordered + list(by_id.values())
    return runs


def _scan_folder(runs_dir: Path) -> list[RunInfo]:
    runs = []
    for path in sorted(runs_dir.iterdir(), reverse=True):
        if not (path / "results.jsonl").exists():
            continue
        info_path = path / RUN_INFO_FILE
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        runs.append(
            RunInfo(
                run_id=path.name,
                path=path,
                # Runs saved before M5 have no "mode"; their max_attempts tells them apart.
                mode=info.get("mode") or ("agent" if "max_attempts" in info else "scan"),
                target_model=info.get("target_model", "unknown"),
                configs=info.get("defense_configs", []),
                attack_count=info.get("attack_count", 0),
                has_benign=(path / "benign.jsonl").exists(),
            )
        )
    return runs


@dataclass
class Run:
    """One loaded scan."""

    info: dict
    results: list[AttackResult]
    benign: list[BenignResult] = field(default_factory=list)

    @property
    def agent_mode(self) -> bool:
        return any(result.attempt > 0 for result in self.results)

    @property
    def configs(self) -> list[str]:
        return defense_configs(self.results)


def load_run(run_dir: Path) -> Run:
    info_path = run_dir / RUN_INFO_FILE
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return Run(info=info, results=load_results(run_dir), benign=load_benign_results(run_dir))


# ------------------------------------------------------------------ overview


def summary_rows(run: Run) -> list[DefenseRow]:
    """ASR (after the agent, for agent runs) and FRR per defense configuration."""
    return defense_summary(run.results, run.benign, run.agent_mode)


def headline(run: Run) -> dict[str, float | None]:
    """The numbers for the stat tiles at the top of the Overview page."""
    rows = {row.config: row for row in summary_rows(run)}
    first, last = (rows.get("none"), rows.get("all"))

    def asr(row: DefenseRow | None) -> float | None:
        return row.asr.asr if row else None

    def frr(row: DefenseRow | None) -> float | None:
        return row.frr.frr if row and row.frr else None

    return {
        "asr_none": asr(first),
        "asr_all": asr(last),
        "frr_none": frr(first),
        "frr_all": frr(last),
    }


def summary_frame(run: Run) -> pd.DataFrame:
    """One row per configuration: ASR, FRR, useful answers (the table view of the charts)."""
    records = []
    for row in summary_rows(run):
        records.append(
            {
                "Configuration": row.config,
                "Short": short_label(row.config),
                "Defenses": len(row.defenses),
                "ASR": row.asr.asr,
                "Attacks through": f"{row.asr.successes}/{row.asr.judged}",
                "Unsure": row.asr.unsure,
                "FRR": row.frr.frr if row.frr else None,
                "Harmless blocked": f"{row.frr.blocked}/{row.frr.asked}" if row.frr else "",
                "Useful": row.frr.usefulness if row.frr else None,
            }
        )
    return pd.DataFrame.from_records(records)


def before_after_frame(run: Run) -> pd.DataFrame:
    """For agent runs: ASR of the seeds as written vs after the agent, per configuration."""
    records = []
    for config in run.configs:
        results = [r for r in run.results if r.defense_config == config]
        for stage, chosen in (
            ("Seeds as written", original_attempts(results)),
            ("After the agent", best_per_seed(results)),
        ):
            stat = compute_asr(chosen)
            records.append(
                {
                    "Configuration": config,
                    "Stage": stage,
                    "ASR": stat.asr,
                    "Attacks through": f"{stat.successes}/{stat.judged}",
                }
            )
    return pd.DataFrame.from_records(records)


def category_frame(run: Run) -> pd.DataFrame:
    """Long table: ASR for each (configuration, category). Feeds the heatmap."""
    results = best_per_seed(run.results) if run.agent_mode else run.results
    records = []
    for config in run.configs:
        for category in CATEGORIES:
            chosen = [r for r in results if r.defense_config == config and r.category == category]
            if not chosen:
                continue
            stat = compute_asr(chosen)
            records.append(
                {
                    "Configuration": short_label(config),
                    "Full configuration": config,
                    "Category": category_name(category),
                    "ASR": stat.asr,
                    "Attacks through": f"{stat.successes}/{stat.judged}",
                }
            )
    return pd.DataFrame.from_records(records)


# ------------------------------------------------------------------- attacks


def attacks_frame(results: list[AttackResult]) -> pd.DataFrame:
    """One row per attack result, for the filterable table on the Attacks page."""
    records = []
    for index, result in enumerate(results):
        blocked = next((t.blocked_by for t in result.transcript if t.blocked_by), None)
        first = result.transcript[0].user if result.transcript else ""
        records.append(
            {
                "row": index,  # position in run.results, to open the transcript
                "Attack": result.attack_id,
                "Seed": result.root_id,
                "Attempt": result.attempt,
                "Strategy": result.strategy or "seed",
                "Category": category_name(result.category),
                "Configuration": result.defense_config,
                "Verdict": result.final_verdict,
                "Decided by": result.decided_by,
                "Blocked by": blocked or "",
                "Turns": len(result.transcript),
                "Language": result.language,
                "First message": first if len(first) <= 120 else first[:117] + "...",
                "Latency (s)": round(result.latency_ms / 1000, 1),
            }
        )
    return pd.DataFrame.from_records(records)


def filter_attacks(
    frame: pd.DataFrame,
    configs: list[str] | None = None,
    categories: list[str] | None = None,
    verdicts: list[str] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Apply the Attacks page filters. An empty list means "don't filter on this"."""
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    if configs:
        mask &= frame["Configuration"].isin(configs)
    if categories:
        mask &= frame["Category"].isin(categories)
    if verdicts:
        mask &= frame["Verdict"].isin(verdicts)
    if search.strip():
        needle = search.strip().lower()
        mask &= frame["Attack"].str.lower().str.contains(needle, regex=False) | frame[
            "First message"
        ].str.lower().str.contains(needle, regex=False)
    return frame[mask]


# ------------------------------------------------------------------- lineage


@dataclass
class LineageNode:
    result: AttackResult
    parent: str | None  # attack id of the parent, None for the seed


def lineage(results: list[AttackResult], config: str, seed_id: str) -> list[LineageNode]:
    """Every attempt descended from one seed under one configuration, in attempt order."""
    chain = [r for r in results if r.defense_config == config and r.root_id == seed_id]
    chain.sort(key=lambda r: (r.attempt, r.attack_id))
    return [LineageNode(result=r, parent=r.parent_id) for r in chain]


def seed_outcomes(results: list[AttackResult], config: str) -> pd.DataFrame:
    """Per seed: did the original succeed, did any variant, and after how many tries.

    "Rescued" = the seed failed as written but the agent found a variant that worked.
    These are the interesting ones to look at on the Lineage page.
    """
    by_seed: dict[str, list[AttackResult]] = {}
    for result in results:
        if result.defense_config == config:
            by_seed.setdefault(result.root_id, []).append(result)
    records = []
    for seed_id, attempts in sorted(by_seed.items()):
        attempts.sort(key=lambda r: r.attempt)
        original = attempts[0].final_verdict == "success"
        winner = next((r for r in attempts if r.final_verdict == "success"), None)
        if original:
            outcome = "worked as written"
        elif winner is not None:
            outcome = "rescued by the agent"
        else:
            outcome = "never worked"
        records.append(
            {
                "Seed": seed_id,
                "Category": category_name(attempts[0].category),
                "Outcome": outcome,
                "Attempts": len(attempts),
                "Winning strategy": (winner.strategy or "seed") if winner else "",
            }
        )
    return pd.DataFrame.from_records(records)


def strategy_frame(results: list[AttackResult], config: str) -> pd.DataFrame:
    """How often each rewrite strategy produced a successful variant."""
    records: dict[str, dict] = {}
    for result in results:
        if result.defense_config != config or result.attempt == 0 or not result.strategy:
            continue
        name = result.strategy.split(":")[0]  # "roleplay:llm" -> "roleplay"
        entry = records.setdefault(name, {"Strategy": name, "Variants": 0, "Succeeded": 0})
        entry["Variants"] += 1
        entry["Succeeded"] += result.final_verdict == "success"
    frame = pd.DataFrame.from_records(list(records.values()))
    if not frame.empty:
        frame["Success rate"] = frame["Succeeded"] / frame["Variants"]
        frame = frame.sort_values(["Succeeded", "Variants"], ascending=False)
    return frame


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def lineage_dot(nodes: list[LineageNode], colours: dict[str, str]) -> str:
    """A Graphviz DOT tree: seed on the left, each rewrite to its right.

    `colours` maps a verdict to a fill colour. Each box also says the verdict in
    words, so colour is never the only signal.
    """
    lines = [
        "digraph lineage {",
        "  rankdir=LR;",
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
        'penwidth=0, margin="0.18,0.1"];',
        '  edge [color="#898781", arrowsize=0.7, fontname="Helvetica", fontsize=10, '
        'fontcolor="#52514e"];',
    ]
    ids = {node.result.attack_id for node in nodes}
    for node in nodes:
        result = node.result
        title = "seed" if result.attempt == 0 else f"attempt {result.attempt}"
        fill = colours.get(result.final_verdict, "#e1e0d9")
        ink = "#ffffff" if result.final_verdict == "success" else "#0b0b0b"
        label = f"{result.attack_id}\\n{title} - {result.final_verdict.upper()}"
        lines.append(
            f'  "{_dot_escape(result.attack_id)}" [label="{label}", fillcolor="{fill}", '
            f'fontcolor="{ink}"];'
        )
        if node.parent and node.parent in ids:
            strategy = (result.strategy or "").split(":")[0]
            lines.append(
                f'  "{_dot_escape(node.parent)}" -> "{_dot_escape(result.attack_id)}" '
                f'[label=" {_dot_escape(strategy)} "];'
            )
    lines.append("}")
    return "\n".join(lines)


# ------------------------------------------------------------------ labelling


def label_progress(labels: dict) -> dict[str, int]:
    """Counts of saved labels by value (skips don't count towards the target)."""
    counts = {"success": 0, "fail": 0, "skip": 0}
    for label in labels.values():
        counts[label.label] = counts.get(label.label, 0) + 1
    return counts
