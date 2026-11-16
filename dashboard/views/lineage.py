"""Lineage: how the adaptive attacker rewrote each seed, attempt by attempt.

Only agent scans (`redrag scan --agent`) have variants. Each seed is tried as
written; if it fails, the agent rewrites it with a strategy (roleplay, rephrase,
encode_output, ...) and tries again, up to max_attempts. The tree shows that
chain, and which rewrite finally got through.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard import common, data, theme
from redrag.eval.charts import short_label
from redrag.eval.labels import plain_goal

loaded = common.current()
if loaded is None:
    st.stop()
info, run = loaded
st.title("Lineage")

if not run.agent_mode:
    st.info(
        f"Scan {info.run_id} has no rewritten variants: lineage needs an adaptive attacker "
        "scan (`redrag scan --agent`).",
        icon=":material/info:",
    )
    agent_runs = [r for r in common.all_runs() if r.mode == "agent"]
    if agent_runs:
        st.markdown("Agent scans you can open:")
        for agent_run in agent_runs[:5]:
            # A callback runs before the sidebar picker is drawn, so it may change it.
            st.button(
                agent_run.label,
                key=f"open-{agent_run.run_id}",
                on_click=common.select_run,
                args=(agent_run.run_id,),
            )
    st.stop()

# --------------------------------------------------------------- controls
c1, c2 = st.columns([1, 2])
config = c1.selectbox(
    "Configuration", run.configs, format_func=lambda c: f"{c} ({short_label(c)})" if "," in c else c
)
outcomes = data.seed_outcomes(run.results, config)
counts = outcomes["Outcome"].value_counts()

tiles = st.columns(3)
tiles[0].metric("Worked as written", int(counts.get("worked as written", 0)))
tiles[1].metric(
    "Rescued by the agent",
    int(counts.get("rescued by the agent", 0)),
    help="Failed as written, but a rewritten variant got through.",
)
tiles[2].metric("Never worked", int(counts.get("never worked", 0)))

choices = ["rescued by the agent", "never worked", "worked as written"]
show = (
    c2.segmented_control("Show seeds that", choices, default=choices[0], key="lineage_outcome")
    or choices[0]
)
seeds = outcomes[outcomes["Outcome"] == show]
if seeds.empty:
    st.info(f"No seeds {show} under this configuration.", icon=":material/info:")
    st.stop()

labels = {
    row["Seed"]: f"{row['Seed']} - {row['Category']} - "
    + (
        f"won with {row['Winning strategy']}"
        if row["Winning strategy"]
        else f"{row['Attempts']} attempt(s)"
    )
    for row in seeds.to_dict("records")
}
seed_id = st.selectbox("Seed", list(labels), format_func=labels.get)

# ------------------------------------------------------------------- tree
nodes = data.lineage(run.results, config, seed_id)
seed_result = nodes[0].result
if seed_result.success_signal is not None:
    st.markdown(f"**Attacker's goal:** {plain_goal(seed_result.success_signal)}")
# Natural size: a two-box chain should not be blown up to the full page width.
st.graphviz_chart(data.lineage_dot(nodes, theme.colours()["verdicts"]), width="content")
st.caption(
    "Each box is one attempt; arrows are labelled with the rewrite strategy. "
    "Filled dark = the attack got through."
)

for node in nodes:
    result = node.result
    title = "Seed as written" if result.attempt == 0 else f"Attempt {result.attempt}"
    strategy = f" - {result.strategy}" if result.strategy else ""
    with st.expander(
        f"{title}: {result.attack_id}{strategy} - {result.final_verdict}",
        expanded=result.final_verdict == "success",
        icon=theme.VERDICT_ICONS.get(result.final_verdict),
    ):
        for number, turn in enumerate(result.transcript, start=1):
            st.markdown(f"**Attacker, turn {number}**")
            st.markdown(theme.escape_md(turn.user))
            st.markdown(f"**Chatbot, turn {number}**" + (
                f" (blocked by {turn.blocked_by})" if turn.blocked_by else ""))  # fmt: skip
            st.markdown(theme.escape_md(turn.answer) or "_(empty answer)_")
        st.caption(f"Decided by {result.decided_by}. {result.notes}")

# ------------------------------------------------------------- strategies
strategies = data.strategy_frame(run.results, config)
if not strategies.empty:
    st.subheader("Which rewrites worked")
    st.caption("Successful variants per rewrite strategy, across all seeds in this configuration.")
    base = alt.Chart(strategies).encode(
        y=alt.Y("Strategy:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("Succeeded:Q", title="Variants that got through", axis=alt.Axis(tickMinStep=1)),
        tooltip=[
            "Strategy:N",
            "Succeeded:Q",
            "Variants:Q",
            alt.Tooltip("Success rate:Q", format=".0%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.65})
    text = base.mark_text(
        align="left", dx=4, fontSize=11, color=theme.colours()["ink_secondary"]
    ).encode(text="Succeeded:Q")
    st.altair_chart((bars + text).properties(height=60 + 34 * len(strategies)), width="stretch")
    with st.expander("Table view", icon=":material/table:"):
        st.dataframe(
            strategies,
            hide_index=True,
            width="stretch",
            column_config={
                "Success rate": st.column_config.NumberColumn(format="percent"),
            },
        )

with st.expander("All seeds in this configuration", icon=":material/list:"):
    st.dataframe(outcomes, hide_index=True, width="stretch")
