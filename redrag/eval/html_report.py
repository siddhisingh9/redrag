"""Writes `runs/<run_id>/report.html`: one self-contained page with charts.

Layout, top to bottom:
1. Headline tiles: ASR with no defenses, ASR with all defenses, FRR with all
   defenses, and the configuration with the best security/usefulness balance.
2. Charts (matplotlib PNGs, embedded, so the file works offline and can be
   emailed): security vs usefulness, ASR and FRR per configuration, and
   ASR per category x configuration.
3. Every table from report.md, converted to HTML, so the two reports never
   disagree. The tables are also the accessible view of each chart.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from redrag.attacks.schema import CATEGORIES, AttackResult
from redrag.eval import charts
from redrag.eval.benign import BenignResult, load_benign_results
from redrag.eval.metrics import asr_table, best_per_seed
from redrag.eval.report import (
    CATEGORY_NAMES,
    best_tradeoff,
    defense_summary,
    load_results,
    load_run_info,
    render_markdown,
)

HTML_REPORT_FILE = "report.html"

STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11, 11, 11, 0.10); --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 8px; }
h3 { font-size: 16px; margin: 24px 0 8px; }
.meta { color: var(--ink-2); margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 30px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12.5px; }
figure { margin: 12px 0 4px; background: var(--surface); border: 1px solid var(--border);
         border-radius: 10px; padding: 12px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { color: var(--ink-2); font-size: 13.5px; margin-top: 8px; }
.table-wrap { overflow-x: auto; margin: 8px 0 12px; }
table { border-collapse: collapse; background: var(--surface); font-size: 13.5px; min-width: 60%; }
th, td { border-bottom: 1px solid var(--grid); padding: 6px 10px;
         text-align: left; vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td { font-variant-numeric: tabular-nums; }
code { font: 12.5px ui-monospace, SFMono-Regular, Consolas, monospace; background: #f0efec;
       padding: 1px 4px; border-radius: 4px; }
details { margin-top: 32px; }
summary { cursor: pointer; font-weight: 600; }
.legend { color: var(--ink-2); font-size: 13px; }
"""


def write_html_report(run_dir: Path, agreement: dict | None = None) -> Path:
    results = load_results(run_dir)
    page = render_html(results, load_run_info(run_dir), agreement, load_benign_results(run_dir))
    path = run_dir / HTML_REPORT_FILE
    path.write_text(page, encoding="utf-8")
    return path


def render_html(
    results: list[AttackResult],
    info: dict,
    agreement: dict | None = None,
    benign: list[BenignResult] | None = None,
) -> str:
    benign = benign or []
    agent_mode = info.get("mode") == "agent" or any(r.attempt > 0 for r in results)
    rows = defense_summary(results, benign, agent_mode)
    by_config = {row.config: row for row in rows}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>RedRAG scan report</title>",
        f"<style>{STYLE}</style></head><body><main>",
        "<h1>RedRAG scan report</h1>",
        f"<p class='meta'>Run <code>{_e(info.get('run_id', '?'))}</code> &middot; "
        f"{_e(info.get('started_at', '?'))} &middot; target model "
        f"<code>{_e(info.get('target_model', '?'))}</code> &middot; "
        f"{len(rows)} defense configuration(s)"
        + (" &middot; adaptive attacker" if agent_mode else "")
        + "</p>",
    ]

    # ---- headline tiles
    tiles = []
    if "none" in by_config:
        tiles.append(_tile("ASR, no defenses", by_config["none"].asr.asr, "attacks that succeeded"))
    if "all" in by_config:
        tiles.append(_tile("ASR, all defenses", by_config["all"].asr.asr, "attacks that succeeded"))
        if by_config["all"].frr:
            tiles.append(
                _tile("FRR, all defenses", by_config["all"].frr.frr, "harmless questions blocked")
            )
    best = best_tradeoff(rows)
    if best is not None:
        tiles.append(
            "<div class='tile'><div class='label'>Best balance (lowest ASR + FRR)</div>"
            f"<div class='value' style='font-size:20px'>{_e(best.config)}</div>"
            f"<div class='note'>ASR {_pct(best.asr.asr)}, FRR {_pct(best.frr.frr)}</div></div>"
        )
    if tiles:
        parts.append(f"<div class='tiles'>{''.join(tiles)}</div>")

    # ---- charts
    legend = (
        "<p class='legend'>Defense abbreviations: IF = input_filter, CS = context_sanitiser, "
        "OF = output_leak_filter, JG = judge_guard.</p>"
    )
    if benign:
        points = [
            (row.config, row.asr.asr, row.frr.frr)
            for row in rows
            if row.asr.asr is not None and row.frr and row.frr.frr is not None
        ]
        parts += [
            "<h2>Security vs usefulness</h2>",
            _figure(
                charts.tradeoff_chart(points),
                "Each dot is one defense configuration. Lower = fewer attacks got through; "
                "further left = fewer harmless questions blocked. The table under "
                "'Security vs usefulness' below has the exact numbers.",
            ),
            legend,
            "<h2>Attack success and false refusals per configuration</h2>",
            _figure(
                charts.asr_frr_bars(
                    [(row.config, row.asr.asr, row.frr.frr if row.frr else None) for row in rows]
                ),
                "Left: share of attacks that succeeded. Right: share of harmless questions "
                "a defense wrongly blocked. Same rows in both panels.",
            ),
        ]
    attacks = best_per_seed(results) if agent_mode else results
    table = asr_table(attacks)
    categories = [c for c in CATEGORIES if c in table]
    configs = [row.config for row in rows]
    if categories and configs:
        values = [[table[c][config].asr for c in categories] for config in configs]
        names = [CATEGORY_NAMES[c].replace(" ", "\n", 1) for c in categories]
        parts += [
            "<h2>Attack success rate by category</h2>",
            _figure(
                charts.category_heatmap(names, configs, values),
                "Darker = more attacks got through. Rows are defense configurations.",
            ),
        ]
        if not benign:
            parts.append(legend)

    # ---- all the detail, shared with report.md
    markdown = render_markdown(results, info, agreement, benign)
    body = markdown.split("\n## ", 1)[1] if "\n## " in markdown else markdown
    parts += [
        "<details open><summary>Detailed tables</summary>",
        markdown_to_html("## " + body),
        "</details>",
        "</main></body></html>",
    ]
    return "\n".join(parts)


def _e(text: object) -> str:
    return html.escape(str(text))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _tile(label: str, value: float | None, note: str) -> str:
    return (
        f"<div class='tile'><div class='label'>{_e(label)}</div>"
        f"<div class='value'>{_pct(value)}</div><div class='note'>{_e(note)}</div></div>"
    )


def _figure(png_base64: str, caption: str) -> str:
    return (
        f"<figure><img alt='{_e(caption)}' src='data:image/png;base64,{png_base64}'>"
        f"<figcaption>{_e(caption)}</figcaption></figure>"
    )


# --------------------------------------------------------------------------
# A tiny Markdown -> HTML converter for the report's own Markdown
# (headings, tables, bullet lists, paragraphs, `code`, **bold**, *italic*).
# --------------------------------------------------------------------------

_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<em>\1</em>", text)
    return text.replace("\\|", "|")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _CELL_SPLIT.split(line.strip().strip("|"))]


def markdown_to_html(markdown: str) -> str:
    out: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            header, rows = block[0], [row for row in block[1:] if not set(row) <= set("|-: ")]
            out.append("<div class='table-wrap'><table><thead><tr>")
            out += [f"<th>{_inline(cell)}</th>" for cell in _cells(header)]
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _cells(row)) + "</tr>"
                )
            out.append("</tbody></table></div>")
            continue
        elif line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)
