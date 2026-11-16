"""Attacks: every result in the scan, filterable; select a row to read the conversation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import common, data, theme
from redrag.eval.labels import plain_goal

loaded = common.current()
if loaded is None:
    st.stop()
info, run = loaded

st.title("Attacks")
frame = data.attacks_frame(run.results)

# Filters: one row above the table.
f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1.2])
configs = f1.multiselect("Configuration", run.configs, placeholder="All configurations")
categories = f2.multiselect(
    "Category", sorted(frame["Category"].unique()), placeholder="All categories"
)
verdicts = f3.multiselect(
    "Verdict", ["success", "unsure", "fail", "error"], placeholder="All verdicts"
)
search = f4.text_input("Search", placeholder="attack id or message text")
shown = data.filter_attacks(frame, configs, categories, verdicts, search)

successes = int((shown["Verdict"] == "success").sum()) if not shown.empty else 0
st.caption(
    f"Showing **{len(shown)}** of {len(frame)} results - **{successes}** got through. "
    "Select a row to read the conversation."
)

event = st.dataframe(
    shown,
    hide_index=True,
    width="stretch",
    height=360,
    on_select="rerun",
    selection_mode="single-row",
    key="attack_table",
    # Strategy only means something in agent runs (plain scans are all "seed").
    column_order=[
        "Attack",
        "Category",
        "Configuration",
        "Verdict",
        "Decided by",
        "Blocked by",
        *(["Strategy"] if run.agent_mode else []),
        "Turns",
        "First message",
    ],  # fmt: skip
    column_config={
        "First message": st.column_config.TextColumn(width="large"),
        "Turns": st.column_config.NumberColumn(width="small"),
    },
)

selected = event.selection.rows if event and event.selection else []
# A filter change can leave a stale selection pointing past the end of the table.
if not selected or selected[0] >= len(shown):
    st.stop()

result = run.results[int(shown.iloc[selected[0]]["row"])]

# ------------------------------------------------------------- transcript
st.divider()
st.subheader(result.attack_id)
facts = st.columns(4)
facts[0].markdown(f"**Verdict**  \n{theme.verdict_text(result.final_verdict)}")
facts[1].markdown(f"**Decided by**  \n{result.decided_by}")
facts[2].markdown(f"**Configuration**  \n{result.defense_config}")
facts[3].markdown(
    f"**Category**  \n{data.category_name(result.category)} ({result.owasp}, {result.language})"
)
if result.success_signal is not None:
    st.markdown(f"**Attacker's goal:** {plain_goal(result.success_signal)}")
if result.attempt > 0:
    st.caption(
        f"Variant {result.attempt} of seed {result.root_id}, rewritten with "
        f"`{result.strategy}` from {result.parent_id}. See the Lineage page for the whole chain."
    )
if result.error:
    st.error(f"This attack ended in an error: {result.error}", icon=":material/error:")

if result.poison_doc_text:
    with st.expander(
        f"Poisoned document slipped into the knowledge base ({result.poison_doc_id})",
        icon=":material/description:",
    ):
        st.markdown(theme.escape_md(result.poison_doc_text))

for number, turn in enumerate(result.transcript, start=1):
    with st.chat_message("user", avatar=":material/person:"):
        st.caption(f"Attacker, turn {number}")
        st.markdown(theme.escape_md(turn.user))
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        note = f"Chatbot, turn {number}"
        if turn.blocked_by:
            note += f" - blocked by {turn.blocked_by}"
        if turn.retrieved_doc_ids:
            note += f" - retrieved: {', '.join(turn.retrieved_doc_ids)}"
        st.caption(note)
        st.markdown(theme.escape_md(turn.answer) or "_(empty answer)_")

st.markdown("**How each judge layer saw it**")
st.dataframe(
    pd.DataFrame.from_records(
        [
            {
                "Layer": verdict.layer,
                "Verdict": verdict.verdict,
                "Score": verdict.score,
                "Reason": verdict.reason,
            }
            for verdict in result.judge_verdicts
        ]
    ),
    hide_index=True,
    width="stretch",
    column_config={"Score": st.column_config.NumberColumn(format="%.2f")},
)
if result.notes:
    st.caption(f"Notes: {result.notes}")
