"""Writes `runs/<run_id>/report.md`. Built in M2; the HTML report with charts comes in M5.

The report is built only from files in the run folder (`run.json` and
`results.jsonl`), so `redrag report <run_dir>` can rebuild it any time, for
example after the judge improves.
"""

from __future__ import annotations

import json
from pathlib import Path

from redrag.attacks.schema import AttackResult
from redrag.eval.metrics import asr_table, compute_asr, defense_configs

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
    lines: list[str] = []

    # ---- header
    lines += [
        "# RedRAG scan report",
        "",
        f"- **Run:** `{info.get('run_id', '?')}`  ({info.get('started_at', '?')})",
        f"- **Target:** {info.get('target_url', '?')}, model `{info.get('target_model', '?')}`",
        f"- **Attacks:** {info.get('attack_count', '?')} per defense configuration, "
        f"{len(results)} runs in total",
        f"- **Defense configurations:** {', '.join(f'`{c}`' for c in configs)}",
        f"- **Judge layers:** {', '.join(info.get('judge_layers', ['rules']))}",
        "",
        "ASR = attack success rate = successful attacks / attacks judged. "
        "Lower is better for the defender. *Unsure* verdicts are not counted as "
        "successes; errors are left out.",
        "",
    ]

    # ---- before vs after table
    table = asr_table(results)
    lines += ["## ASR by category and defense configuration", ""]
    lines.append("| Category | " + " | ".join(f"`{c}`" for c in configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for row, stats in table.items():
        cells = [stats[config].format() for config in configs]
        lines.append(f"| {CATEGORY_NAMES.get(row, row)} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- which defenses blocked attacks
    lines += ["## What stopped the attacks", ""]
    lines.append(
        "| Defense configuration | Blocked by input_filter | Blocked by output_leak_filter "
        "| Errors |"
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
            r for r in results if r.defense_config == strongest and r.final_verdict == "success"
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


def _blocked_by(result: AttackResult, defense: str) -> bool:
    return any(turn.blocked_by == defense for turn in result.transcript)


def _cell(text: str, limit: int = 120) -> str:
    """Make text safe for a Markdown table cell: one line, no pipes, shortened."""
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text
