"""Writes `runs/<run_id>/report.md`. Built in M2; the HTML report with charts comes in M5.

The report is built only from files in the run folder (`run.json` and
`results.jsonl`), so `redrag report <run_dir>` can rebuild it any time, for
example after the judge improves.
"""

from __future__ import annotations

import json
from pathlib import Path

from redrag.attacks.schema import AttackResult
from redrag.eval.metrics import (
    asr_table,
    best_per_seed,
    compute_asr,
    defense_configs,
    original_attempts,
)

RESULTS_FILE = "results.jsonl"
RUN_INFO_FILE = "run.json"
REPORT_FILE = "report.md"

CATEGORY_NAMES = {
    "direct_injection": "Direct injection",
    "indirect_injection": "Indirect injection",
    "system_prompt_extraction": "System prompt extraction",
    "data_exfiltration": "Data exfiltration",
    "multi_turn_escalation": "Multi-turn escalation",
    "overall": "**Overall**",
}


def load_results(run_dir: Path) -> list[AttackResult]:
    path = run_dir / RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {RESULTS_FILE} in {run_dir}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AttackResult.model_validate_json(line) for line in lines if line.strip()]


def load_run_info(run_dir: Path) -> dict:
    path = run_dir / RUN_INFO_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown_report(run_dir: Path) -> Path:
    """Build report.md from the run folder and return its path."""
    results = load_results(run_dir)
    info = load_run_info(run_dir)
    report = render_markdown(results, info)
    path = run_dir / REPORT_FILE
    path.write_text(report, encoding="utf-8")
    return path


def render_markdown(results: list[AttackResult], info: dict) -> str:
    configs = defense_configs(results)
    agent_mode = info.get("mode") == "agent" or any(r.attempt > 0 for r in results)
    # In agent mode, "best" = one result per seed (did ANY variant break through?).
    best = best_per_seed(results) if agent_mode else results
    lines: list[str] = []

    # ---- header
    lines += [
        "# RedRAG scan report",
        "",
        f"- **Run:** `{info.get('run_id', '?')}`  ({info.get('started_at', '?')})",
        f"- **Target:** {info.get('target_url', '?')}, model `{info.get('target_model', '?')}`",
        f"- **Attacks:** {info.get('attack_count', '?')} seeds per defense configuration, "
        f"{len(results)} attempts in total",
        f"- **Defense configurations:** {', '.join(f'`{c}`' for c in configs)}",
        f"- **Judge layers:** {', '.join(info.get('judge_layers', ['rules']))}",
    ]
    if agent_mode:
        lines.append(
            f"- **Mode:** adaptive attacker agent, mutator `{info.get('mutator', '?')}`, "
            f"up to {info.get('max_attempts', '?')} tries per seed (original included)"
        )
    lines += [
        "",
        "ASR = attack success rate = successful attacks / attacks judged. "
        "Lower is better for the defender. *Unsure* verdicts are not counted as "
        "successes; errors are left out.",
        "",
    ]

    # ---- before vs after table
    lines += ["## ASR by category and defense configuration", ""]
    if agent_mode:
        lines.append(
            "*original* = the seeds as written; *+ agent* = a seed counts as broken if any "
            "of its variants succeeded."
        )
        lines.append("")
        original_table = asr_table(original_attempts(results))
        agent_table = asr_table(best)
        headers = []
        for config in configs:
            headers += [f"`{config}` original", f"`{config}` + agent"]
        lines.append("| Category | " + " | ".join(headers) + " |")
        lines.append("|---|" + "---|" * len(headers))
        for row in agent_table:
            cells = []
            for config in configs:
                cells += [original_table[row][config].format(), agent_table[row][config].format()]
            lines.append(f"| {CATEGORY_NAMES.get(row, row)} | " + " | ".join(cells) + " |")
    else:
        table = asr_table(results)
        lines.append("| Category | " + " | ".join(f"`{c}`" for c in configs) + " |")
        lines.append("|---|" + "---|" * len(configs))
        for row, stats in table.items():
            cells = [stats[config].format() for config in configs]
            lines.append(f"| {CATEGORY_NAMES.get(row, row)} | " + " | ".join(cells) + " |")
    lines.append("")

    if agent_mode:
        lines += _agent_section(results, configs)

    # ---- which defenses blocked attacks
    lines += ["## What stopped the attacks", ""]
    lines.append(
        "| Defense configuration | Attempts blocked by input_filter "
        "| Attempts blocked by output_leak_filter | Errors |"
    )
    lines.append("|---|---|---|---|")
    for config in configs:
        config_results = [r for r in results if r.defense_config == config]
        blocked_in = sum(_blocked_by(r, "input_filter") for r in config_results)
        blocked_out = sum(_blocked_by(r, "output_leak_filter") for r in config_results)
        errors = compute_asr(config_results).errors
        lines.append(f"| `{config}` | {blocked_in} | {blocked_out} | {errors} |")
    lines.append("")
    lines.append(
        "*context_sanitiser* doesn't block answers (it cleans the retrieved text), so its "
        "effect only shows up as a lower ASR."
    )
    lines.append("")

    # ---- attacks that beat the strongest configuration
    strongest = configs[-1] if configs else None
    if strongest and strongest != "none":
        survivors = [
            r for r in best if r.defense_config == strongest and r.final_verdict == "success"
        ]
        lines += [f"## Attacks that still succeeded with `{strongest}`", ""]
        if survivors:
            lines += [
                "| Attack | Category | Why the judge said success | Last answer |",
                "|---|---|---|---|",
            ]
            for r in survivors:
                lines.append(
                    f"| `{r.attack_id}` | {r.category} | {_cell(r.judge_verdicts[0].reason)} "
                    f"| {_cell(r.transcript[-1].answer if r.transcript else '')} |"
                )
        else:
            lines.append("None. Every attack was stopped.")
        lines.append("")

    # ---- full results
    lines += ["## All results", ""]
    lines += [
        "| Attack | Category | Defenses | Verdict | Judge reason | Blocked by |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        blocked = ", ".join(sorted({t.blocked_by for t in r.transcript if t.blocked_by})) or ""
        reason = r.error or (r.judge_verdicts[0].reason if r.judge_verdicts else "")
        lines.append(
            f"| `{r.attack_id}` | {r.category} | `{r.defense_config}` | {r.final_verdict} "
            f"| {_cell(reason)} | {blocked} |"
        )
    lines.append("")
    lines.append(
        "All secrets in this report are fake canaries planted in the test app. "
        "Full transcripts are in `results.jsonl`."
    )
    return "\n".join(lines) + "\n"


def _agent_section(results: list[AttackResult], configs: list[str]) -> list[str]:
    """Which seeds failed as written but were broken by a variant, and how."""
    lines = ["## Adaptive attacker", ""]
    lines += [
        "| Defense configuration | Seeds | Broken as written | Failed as written "
        "| ...then broken by a variant | Variants tried |",
        "|---|---|---|---|---|---|",
    ]
    breakthroughs: list[AttackResult] = []
    for config in configs:
        config_results = [r for r in results if r.defense_config == config]
        originals = [r for r in config_results if r.attempt == 0]
        best = {r.root_id: r for r in best_per_seed(config_results)}
        broke_original = [r for r in originals if r.final_verdict == "success"]
        failed_original = [r for r in originals if r.final_verdict != "success"]
        rescued = [
            best[r.root_id] for r in failed_original if best[r.root_id].final_verdict == "success"
        ]
        breakthroughs += rescued
        variants = sum(1 for r in config_results if r.attempt > 0)
        lines.append(
            f"| `{config}` | {len(originals)} | {len(broke_original)} | {len(failed_original)} "
            f"| {len(rescued)} | {variants} |"
        )
    lines.append("")

    lines += ["### Seeds that failed as written but a variant got through", ""]
    if not breakthroughs:
        lines += ["None.", ""]
        return lines

    by_id = {(r.defense_config, r.attack_id): r for r in results}
    lines += [
        "| Seed | Defenses | Winning attempt | Lineage (strategy) | Winning attack |",
        "|---|---|---|---|---|",
    ]
    for winner in breakthroughs:
        lines.append(
            f"| `{winner.root_id}` | `{winner.defense_config}` | {winner.attempt} "
            f"| {_cell(_lineage(winner, by_id), 160)} | {_cell(_attack_text(winner), 160)} |"
        )
    lines.append("")
    return lines


def _lineage(result: AttackResult, by_id: dict[tuple[str, str], AttackResult]) -> str:
    """e.g. "dir-001 → dir-001.v1 (rephrase:llm) → dir-001.v2 (roleplay:template)"."""
    chain: list[AttackResult] = [result]
    while chain[0].parent_id and (result.defense_config, chain[0].parent_id) in by_id:
        chain.insert(0, by_id[(result.defense_config, chain[0].parent_id)])
    steps = []
    for step in chain:
        steps.append(f"{step.attack_id} ({step.strategy})" if step.strategy else step.attack_id)
    return " → ".join(steps)


def _attack_text(result: AttackResult) -> str:
    """What the winning attack actually sent (the hidden instruction, for poison attacks)."""
    if result.category == "indirect_injection" and result.poison_doc_text:
        return f"poison doc: {result.poison_doc_text}"
    return " / ".join(turn.user for turn in result.transcript)


def _blocked_by(result: AttackResult, defense: str) -> bool:
    return any(turn.blocked_by == defense for turn in result.transcript)


def _cell(text: str, limit: int = 120) -> str:
    """Make text safe for a Markdown table cell: one line, no pipes, shortened."""
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text
