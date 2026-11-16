"""Build the example scans for the hosted dashboard demo.

    python scripts/build_demo.py

Writes demo/runs/<run id>/ for each scan in SCANS: its results, run info and HTML
report (shown on the dashboard's Report page), plus demo/runs/index.json with
their order and titles for the scan picker. The dashboard shows these when there
are no scans of your own, which is how the hosted demo on Streamlit Community
Cloud runs.

Run it again after scans you want to show, then commit demo/. Everything in these
scans is fake (canary strings, a fictional college), so it is safe to publish.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redrag.config import load_config, resolve_path  # noqa: E402
from redrag.eval.benign import load_benign_results  # noqa: E402
from redrag.eval.html_report import HTML_REPORT_FILE, render_html  # noqa: E402
from redrag.eval.report import load_results, load_run_info  # noqa: E402
from redrag.factory import load_agreement  # noqa: E402

DEMO_RUNS = ROOT / "demo" / "runs"


@dataclass
class Scan:
    run_id: str  # folder name in demo/runs (the newest part, for merged scans)
    title: str  # shown in the dashboard's scan picker
    # Scans that were interrupted and finished in a second run: list every part,
    # oldest first. For each seed, the newest part that ran it wins.
    parts: list[str] = field(default_factory=list)


SCANS = [
    # 20 attacks + 50 harmless questions against all 16 defense combinations.
    Scan("20261116-034159", "Every defense combination"),
    # 55 seeds; failed ones rewritten by the agent (up to 4 tries); rule-based defenses.
    Scan("20261115-224721", "Adaptive attacker vs rule-based defenses"),
    # The same against the fine-tuned classifier used as a defense (judge_guard).
    Scan(
        "20261116-153047",
        "Adaptive attacker vs the judge guard",
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


def main() -> None:
    config = load_config()
    runs_dir = resolve_path(config.output.runs_dir)
    agreement = load_agreement(config)
    for scan in SCANS:
        run_dir = copy_scan(scan, runs_dir)
        page = render_html(
            load_results(run_dir), load_run_info(run_dir), agreement, load_benign_results(run_dir)
        )
        (run_dir / HTML_REPORT_FILE).write_text(page, encoding="utf-8")
        print(f"{scan.title}: {run_dir.relative_to(ROOT)}")
    # Order and titles for the dashboard's scan picker (the first one opens by default).
    index = [{"run_id": scan.run_id, "title": scan.title} for scan in SCANS]
    (DEMO_RUNS / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
