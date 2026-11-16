"""RedRAG dashboard (Streamlit). Start it from the project folder with:

    streamlit run dashboard/app.py

Pages (spec 6.10, plus the full report):
- Overview: headline ASR/FRR, before-vs-after defenses, per-category heatmap.
- Attacks:  filterable table of every attack result; click one to read the conversation.
- Lineage:  for agent scans, how each seed was rewritten, attempt by attempt.
- Report:   the scan's full HTML report, viewable and downloadable.
- Label:    hand-label results as success/fail (same file as `redrag label`).

It only reads saved scans in `runs/` (and writes labels on the Label page). It
never attacks anything, so it is safe to leave open while scans run. With no
scans of your own it shows the bundled examples in `demo/runs/`, read-only
(this is also how the hosted demo runs; see common.demo_mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts dashboard/ on the import path, not the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard import common, theme  # noqa: E402

st.set_page_config(page_title="RedRAG", page_icon=":material/shield:", layout="wide")
theme.apply()

# Page paths are relative to this file (Streamlit resolves them from the main script).
demo = common.demo_mode()
pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/attacks.py", title="Attacks", icon=":material/table_rows:"),
    st.Page("views/lineage.py", title="Lineage", icon=":material/account_tree:"),
    st.Page("views/reports.py", title="Report", icon=":material/description:"),
]
if not demo:  # labelling writes files, so the read-only demo leaves it out
    pages.append(st.Page("views/label.py", title="Label", icon=":material/rule:"))
page = st.navigation(pages)

with st.sidebar:
    st.markdown("**RedRAG**  \nRed-teaming results for RAG apps")
    if demo:
        st.caption(
            "Example scans of the bundled target app (llama3.2:3b). "
            "[Source on GitHub](https://github.com/siddhisingh9/redrag)"
        )
if page.title != "Label":  # labelling works across all runs, not one
    common.run_picker()

page.run()
