"""Build the public demo: example scans for the hosted dashboard, reports for GitHub Pages.

    python scripts/build_demo.py

Writes:
- demo/runs/<run id>/  copies of the scans listed in SCANS. The dashboard shows
                       these when there are no scans of your own, which is how the
                       hosted demo on Streamlit Community Cloud runs.
- demo/runs/index.json their order and titles for the dashboard's scan picker.
- docs/reports/*.html  each scan's self-contained HTML report.
- docs/index.html      a small landing page linking the reports and the dashboard,
                       served by GitHub Pages (Settings -> Pages -> main, /docs).

Run it again after scans you want to show, then commit demo/ and docs/.
Everything in these scans is fake (canary strings, a fictional college), so it is
safe to publish.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redrag.config import load_config, resolve_path  # noqa: E402
from redrag.eval.benign import load_benign_results  # noqa: E402
from redrag.eval.html_report import render_html  # noqa: E402
from redrag.eval.report import load_results, load_run_info  # noqa: E402
from redrag.factory import load_agreement  # noqa: E402

DEMO_RUNS = ROOT / "demo" / "runs"
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/siddhisingh9/redrag"
DASHBOARD_URL = "https://redrag.streamlit.app"


@dataclass
class Scan:
    run_id: str  # folder name in demo/runs (the newest part, for merged scans)
    slug: str  # file name of the report in docs/reports
    title: str
    blurb: str
    # Scans that were interrupted and finished in a second run: list every part,
    # oldest first. For each seed, the newest part that ran it wins.
    parts: list[str] = field(default_factory=list)


SCANS = [
    Scan(
        "20261116-034159",
        "defense-combinations",
        "Every defense combination",
        "20 attacks and 50 harmless questions against all 16 on/off combinations of "
        "the four defenses: attack success rate vs false-refusal rate.",
    ),
    Scan(
        "20261115-224721",
        "agent-vs-rule-defenses",
        "Adaptive attacker vs rule-based defenses",
        "55 seed attacks; failed ones are rewritten by the LangGraph agent (up to 4 tries) "
        "against the input filter, context sanitiser and output leak filter.",
    ),
    Scan(
        "20261116-153047",
        "agent-vs-judge-guard",
        "Adaptive attacker vs the judge guard",
        "The same 55 seeds and rewrites against the fine-tuned classifier used as a "
        "defense (judge_guard).",
        parts=["20261116-114853", "20261116-153047"],
    ),
]


def copy_scan(scan: Scan, runs_dir: Path) -> Path:
    """Copy (or merge) a scan's data files into demo/runs/<run id>."""
    target = DEMO_RUNS / scan.run_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    parts = scan.parts or [scan.run_id]

    # Results: for each seed, keep the attempts from the newest part that ran it.
    by_seed: dict[str, list[str]] = {}
    for part in parts:
        seen: dict[str, list[str]] = {}
        for line in (runs_dir / part / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                seen.setdefault(row.get("seed_id") or row["attack_id"], []).append(line)
        by_seed.update(seen)
    lines = [line for seed in by_seed.values() for line in seed]
    (target / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    info = json.loads((runs_dir / parts[-1] / "run.json").read_text(encoding="utf-8"))
    if len(parts) > 1:
        first = json.loads((runs_dir / parts[0] / "run.json").read_text(encoding="utf-8"))
        info["started_at"] = first.get("started_at", info.get("started_at"))
        info["attack_ids"] = sorted(by_seed)
        info["attack_count"] = len(by_seed)
        info["merged_from"] = parts
        info["note"] = (
            "The first scan was interrupted; the categories it had not reached were "
            "scanned again and the two were merged (newest result per seed)."
        )
    (target / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    benign = runs_dir / parts[-1] / "benign.jsonl"
    if benign.exists() and benign.stat().st_size:
        shutil.copyfile(benign, target / "benign.jsonl")
    return target


def landing_page(scans: list[Scan]) -> str:
    items = "\n".join(
        f'<li><a href="reports/{s.slug}.html">{html.escape(s.title)}</a>'
        f"<p>{html.escape(s.blurb)}</p></li>"
        for s in scans
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RedRAG</title>
<style>
:root {{ --bg: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --line: #e1e0d9; --link: #256abf; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --line: #2c2c2a; --link: #6da7ec; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--ink);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 720px; margin: 0 auto; padding: 56px 16px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
h2 {{ font-size: 16px; margin: 40px 0 8px; color: var(--ink-2); font-weight: 600; }}
p {{ margin: 0 0 12px; color: var(--ink-2); }}
a {{ color: var(--link); }}
ul {{ list-style: none; padding: 0; margin: 0; }}
li {{ border-top: 1px solid var(--line); padding: 14px 0; }}
li a {{ font-weight: 600; }}
li p {{ margin: 4px 0 0; font-size: 15px; }}
</style>
</head>
<body>
<main>
<h1>RedRAG</h1>
<p>Automated red-teaming for RAG chatbots: prompt injection, data exfiltration and
system-prompt extraction attacks, a three-layer judge, and a measurement of what each
defense costs in false refusals. All secrets in the target app are fake.</p>
<p><a href="{DASHBOARD_URL}">Open the dashboard</a> &middot;
<a href="{REPO_URL}">Source on GitHub</a></p>
<h2>Scan reports</h2>
<ul>
{items}
</ul>
</main>
</body>
</html>
"""


def main() -> None:
    config = load_config()
    runs_dir = resolve_path(config.output.runs_dir)
    agreement = load_agreement(config)
    (DOCS / "reports").mkdir(parents=True, exist_ok=True)
    for scan in SCANS:
        run_dir = copy_scan(scan, runs_dir)
        page = render_html(
            load_results(run_dir), load_run_info(run_dir), agreement, load_benign_results(run_dir)
        )
        (DOCS / "reports" / f"{scan.slug}.html").write_text(page, encoding="utf-8")
        print(f"{scan.slug}: {run_dir.relative_to(ROOT)}")
    # Order and titles for the dashboard's scan picker (the first one opens by default).
    index = [{"run_id": scan.run_id, "title": scan.title} for scan in SCANS]
    (DEMO_RUNS / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (DOCS / "index.html").write_text(landing_page(SCANS), encoding="utf-8")
    print("docs/index.html")


if __name__ == "__main__":
    main()
