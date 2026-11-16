"""Writes `runs/<run_id>/report.md` (and, via html_report.py, `report.html`).

The report is built only from files in the run folder (`run.json` and
`results.jsonl`), so `redrag report <run_dir>` can rebuild it any time, for
example after the judge improves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from redrag.attacks.schema import AttackResult
from redrag.eval.benign import BenignResult, answer_leaks, leaky_questions, load_benign_results
from redrag.eval.metrics import (
    ASRStat,
    FRRStat,
    asr_table,
    best_per_seed,
    compute_asr,
    defense_configs,
    frr_by_config,
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


def write_markdown_report(run_dir: Path, agreement: dict | None = None) -> Path:
    """Build report.md from the run folder and return its path.

    `agreement` is the latest judge-agreement result (from `redrag judge-eval`);
    if given, its table is included in the report.
    """
    results = load_results(run_dir)
    info = load_run_info(run_dir)
    report = render_markdown(results, info, agreement, load_benign_results(run_dir))
    path = run_dir / REPORT_FILE
    path.write_text(report, encoding="utf-8")
    return path


@dataclass
class DefenseRow:
    """One defense configuration's headline numbers, for the trade-off table and charts."""

    config: str
    defenses: list[str]
    asr: ASRStat
    frr: FRRStat | None


def defense_summary(
    results: list[AttackResult], benign: list[BenignResult], agent_mode: bool
) -> list[DefenseRow]:
    """ASR (after the agent, in agent runs) and FRR for each defense configuration."""
    attacks = best_per_seed(results) if agent_mode else results
    frr = frr_by_config(benign)
    rows = []
    for config in defense_configs(results) or list(frr):
        config_results = [r for r in attacks if r.defense_config == config]
        defenses = next((r.defenses for r in config_results if r.defenses), [])
        if not defenses:
            defenses = next((b.defenses for b in benign if b.defense_config == config), [])
        rows.append(DefenseRow(config, defenses, compute_asr(config_results), frr.get(config)))
    return rows


def best_tradeoff(rows: list[DefenseRow]) -> DefenseRow | None:
    """The configuration with the lowest ASR + FRR (ties: fewer defenses)."""
    scored = [r for r in rows if r.asr.asr is not None and r.frr and r.frr.frr is not None]
    if not scored:
        return None
    return min(scored, key=lambda r: (r.asr.asr + r.frr.frr, len(r.defenses)))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _tradeoff_section(rows: list[DefenseRow]) -> list[str]:
    lines = [
        "## Security vs usefulness",
        "",
        "For each defense configuration: how many attacks still got through (ASR, lower "
        "is better), how many harmless questions a defense wrongly blocked (FRR, lower is "
        "better), and how many harmless questions got a useful answer (higher is better).",
        "",
        "| Configuration | Defenses on | ASR | FRR | Useful answers |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        frr = row.frr.format() if row.frr else "n/a"
        useful = _pct(row.frr.usefulness) if row.frr else "n/a"
        defenses = ", ".join(row.defenses) or "none"
        lines.append(f"| `{row.config}` | {defenses} | {row.asr.format()} | {frr} | {useful} |")
    best = best_tradeoff(rows)
    if best is not None:
        lines += [
            "",
            f"Lowest ASR + FRR: `{best.config}` (ASR {_pct(best.asr.asr)}, "
            f"FRR {_pct(best.frr.frr)}).",
        ]
    return [*lines, ""]


def _leaky_section(benign: list[BenignResult]) -> list[str]:
    """Harmless questions that made the model leak a secret with no attack."""
    leaky = leaky_questions(benign)
    if not leaky:
        return []
    lines = [
        "## Harmless questions that leaked a secret",
        "",
        "These harmless questions made the model reveal a secret (a canary or the "
        "restricted student's data) **with no attack at all**. Blocking them is the right "
        "call, so they are left out of FRR and useful answers above.",
        "",
        "| Question | Leaked under | Blocked under (by) |",
        "|---|---|---|",
    ]
    for question_id in sorted(leaky):
        rows = [r for r in benign if r.question_id == question_id and r.error is None]
        leaked = [
            f"`{r.defense_config}`"
            for r in rows
            if r.blocked_by is None and answer_leaks(r.answer, r.question)
        ]
        blocked = [f"`{r.defense_config}` ({r.blocked_by})" for r in rows if r.blocked_by]
        lines.append(
            f"| {_cell(rows[0].question)} | {', '.join(leaked)} | {', '.join(blocked) or '-'} |"
        )
    return [*lines, ""]


def _false_refusal_section(benign: list[BenignResult]) -> list[str]:
    """Which harmless questions were blocked, by which defense, under which configurations."""
    leaky = leaky_questions(benign)  # listed in their own section; blocking them is right
    blocked: dict[str, dict] = {}
    for result in benign:
        if result.blocked_by is None or result.question_id in leaky:
            continue
        entry = blocked.setdefault(
            result.question_id,
            {"question": result.question, "tricky": result.tricky, "by": set(), "configs": []},
        )
        entry["by"].add(result.blocked_by)
        entry["configs"].append(result.defense_config)
    lines = ["## Harmless questions that were blocked", ""]
    if not blocked:
        return [*lines, "None: no defense blocked a harmless question.", ""]
    lines += [
        "*Tricky* questions sound like attacks but are harmless.",
        "",
        "| Question | Tricky | Blocked by | Under configurations |",
        "|---|---|---|---|",
    ]
    for question_id in sorted(blocked):
        entry = blocked[question_id]
        configs = ", ".join(f"`{c}`" for c in entry["configs"])
        lines.append(
            f"| {_cell(entry['question'])} | {'yes' if entry['tricky'] else ''} "
            f"| {', '.join(sorted(entry['by']))} | {configs} |"
        )
    return [*lines, ""]


def render_markdown(
    results: list[AttackResult],
    info: dict,
    agreement: dict | None = None,
    benign: list[BenignResult] | None = None,
) -> str:
    benign = benign or []
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

    if benign:
        lines += _tradeoff_section(defense_summary(results, benign, agent_mode))
        lines += _false_refusal_section(benign)
        lines += _leaky_section(benign)

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

    # ---- which judge layer settled each attack
    lines += _judging_section(results)
    if agreement is not None:
        from redrag.eval.judge_eval import render_agreement_markdown

        lines += [render_agreement_markdown(agreement), ""]

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


def _judging_section(results: list[AttackResult]) -> list[str]:
    """How many attacks each judge layer decided (rules -> classifier -> Gemini)."""
    order = ["rules", "classifier", "gemini", "none"]
    counts = {name: 0 for name in order}
    for result in results:
        if result.final_verdict != "error":
            counts[result.decided_by if result.decided_by in counts else "none"] += 1
    labels = {
        "rules": "Rules",
        "classifier": "Fine-tuned classifier",
        "gemini": "Gemini",
        "none": "Nobody (left *unsure*)",
    }
    lines = ["## Which judge decided", "", "| Judge layer | Attacks decided |", "|---|---|"]
    lines += [f"| {labels[name]} | {counts[name]} |" for name in order]
    lines.append("")
    return lines


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
