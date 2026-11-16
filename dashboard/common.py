"""Shared by every page: settings, the run picker, cached loading.

The run picker lives in the sidebar and is drawn by `app.py` on every page, so
the chosen run stays selected when you move between Overview, Attacks and
Lineage. Loaded runs are cached and reloaded only when results.jsonl changes
(e.g. after `redrag report --rejudge`).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.data import Run, RunInfo, list_runs, load_run
from redrag.config import Config, load_config, resolve_path

RUN_KEY = "run_id"  # session-state key of the run picker


def config() -> Config:
    return load_config()


def runs_dir() -> Path:
    return resolve_path(config().output.runs_dir)


def all_runs() -> list[RunInfo]:
    return list_runs(runs_dir())


@st.cache_resource(show_spinner="Loading the scan...", max_entries=8)
def _load(path: str, modified: float) -> Run:
    # `modified` is only part of the cache key: a re-judged run is reloaded.
    return load_run(Path(path))


def load(info: RunInfo) -> Run:
    results = info.path / "results.jsonl"
    benign = info.path / "benign.jsonl"
    modified = results.stat().st_mtime + (benign.stat().st_mtime if benign.exists() else 0)
    return _load(str(info.path), modified)


def run_picker() -> None:
    """Sidebar select box with every saved run (newest first)."""
    runs = all_runs()
    if not runs:
        return
    ids = [run.run_id for run in runs]
    labels = {run.run_id: run.label for run in runs}
    if st.session_state.get(RUN_KEY) not in ids:
        # First load: a link like ?run=20261116-034159 opens that scan, else the newest.
        wanted = st.query_params.get("run")
        st.session_state[RUN_KEY] = wanted if wanted in ids else ids[0]
    with st.sidebar:
        st.selectbox(
            "Scan",
            ids,
            key=RUN_KEY,
            format_func=lambda run_id: labels[run_id],
            help="Saved scans from the runs folder, newest first.",
        )
        st.caption(f"Runs folder: `{runs_dir()}`")


def current() -> tuple[RunInfo, Run] | None:
    """The run chosen in the sidebar, loaded. Shows a hint and returns None if there are none."""
    runs = {run.run_id: run for run in all_runs()}
    if not runs:
        st.info(
            "No saved scans yet. Run one first, for example "
            "`redrag scan --mock-target` (offline, a few seconds), then reload this page.",
            icon=":material/info:",
        )
        return None
    info = runs.get(st.session_state.get(RUN_KEY)) or next(iter(runs.values()))
    return info, load(info)


def select_run(run_id: str) -> None:
    """Switch the sidebar picker to another run. Call it from a button's on_click."""
    st.session_state[RUN_KEY] = run_id


def agreement() -> dict | None:
    """The judge-agreement table saved by `redrag judge-eval`, if there is one."""
    from redrag.factory import load_agreement

    return load_agreement(config())
