"""Overview: how many attacks got through, with and without defenses, and at what cost.

Charts follow the report's rules: one blue for data, magnitude as one blue ramp,
never two y-scales on one chart (ASR and FRR get separate panels), hover
tooltips on every mark, and every chart has a table view with the same numbers.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common, data, theme
from redrag.eval.charts import short_label
from redrag.eval.report import best_tradeoff

loaded = common.current()
if loaded is None:
    st.stop()
info, run = loaded
ink = theme.colours()

st.title("Overview")
judge = " -> ".join(run.info.get("judge_layers", [])) or "rules"
st.caption(
    f"Scan **{info.run_id}** against **{info.target_model}** - "
    f"{len({r.root_id for r in run.results})} attacks"
    + (f" + {len({b.question_id for b in run.benign})} harmless questions" if run.benign else "")
    + f" - {len(run.configs)} defense configuration(s) - judged by {judge}"
    + (" - adaptive attacker (agent) run" if run.agent_mode else "")
)

# ------------------------------------------------------------------ stat tiles
numbers = data.headline(run)
rows = data.summary_rows(run)
best = best_tradeoff(rows)
tiles = st.columns(4)


def tile(column, *args, **kwargs) -> None:
    """A stat tile with a hairline border."""
    column.metric(*args, border=True, **kwargs)


if run.agent_mode:
    # Agent runs: the question is how much the rewrites added, per the most defended config.
    stages = data.before_after_frame(run)
    focus = "all" if "all" in run.configs else run.configs[-1]
    picked = stages[stages["Configuration"] == focus]
    asr = dict(zip(picked["Stage"], picked["ASR"], strict=True))
    before, after = asr.get("Seeds as written"), asr.get("After the agent")
    outcomes = data.seed_outcomes(run.results, focus)
    tile(
        tiles[0],
        "ASR as written",
        theme.pct(before),
        help=f"Share of seed attacks that got through on their first try ({focus}).",
    )
    tile(
        tiles[1],
        "ASR after the agent",
        theme.pct(after),
        f"{((after or 0) - (before or 0)) * 100:+.0f} pts",
        delta_color="inverse",  # more attacks through is bad for the defender
        help=f"A seed counts as broken if any of its rewritten variants got through ({focus}).",
    )
    tile(
        tiles[2],
        "Seeds rescued by the agent",
        int((outcomes["Outcome"] == "rescued by the agent").sum()),
        help="Seeds that failed as written, but a rewritten variant got through.",
    )
    tile(tiles[3], "Variants tried", sum(r.attempt > 0 for r in run.results))
else:
    tile(
        tiles[0],
        "ASR, no defenses",
        theme.pct(numbers["asr_none"]),
        help="Attack success rate: share of attacks that got through with every defense off.",
    )
    delta = None
    if numbers["asr_none"] is not None and numbers["asr_all"] is not None:
        delta = f"{(numbers['asr_all'] - numbers['asr_none']) * 100:+.0f} pts"
    tile(
        tiles[1],
        "ASR, all defenses",
        theme.pct(numbers["asr_all"]),
        delta,
        delta_color="inverse",  # lower is better for the defender
        help="Same attacks with every defense on. Change in percentage points vs no defenses.",
    )
    if run.benign:
        tile(
            tiles[2],
            "FRR, all defenses",
            theme.pct(numbers["frr_all"]),
            help="False-refusal rate: harmless questions a defense wrongly blocked.",
        )
        tile(
            tiles[3],
            "Best balance",
            short_label(best.config) if best else "n/a",
            f"ASR {theme.pct(best.asr.asr)}, FRR {theme.pct(best.frr.frr)}" if best else None,
            delta_color="off",
            delta_arrow="off",
            help="The configuration with the lowest ASR + FRR. IF = input filter, "
            "CS = context sanitiser, OF = output leak filter, JG = judge guard.",
        )
    else:
        tile(tiles[2], "Attacks", len({r.root_id for r in run.results}))
        tile(tiles[3], "Configurations", len(run.configs))


# ---------------------------------------------------------------------- charts
def bar_panel(frame: pd.DataFrame, value: str, title: str, detail: str) -> alt.Chart:
    """Horizontal bars, one per configuration, in the scan's order, with a value label."""
    base = alt.Chart(frame, title=alt.Title(title, anchor="start", fontSize=13)).encode(
        y=alt.Y("Short:N", sort=None, title=None),
        x=alt.X(
            f"{value}:Q",
            title=None,
            scale=alt.Scale(domain=[0, max(0.1, float(frame[value].max() or 0)) * 1.15]),
            axis=alt.Axis(format="%", tickCount=4, grid=True),
        ),
        tooltip=[
            alt.Tooltip("Configuration:N"),
            alt.Tooltip(f"{value}:Q", format=".0%"),
            alt.Tooltip(f"{detail}:N"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.65})
    labels = base.mark_text(align="left", dx=4, fontSize=11, color=ink["ink_secondary"]).encode(
        text=alt.Text(f"{value}:Q", format=".0%")
    )
    return (bars + labels).properties(height=50 + 26 * len(frame))


summary = data.summary_frame(run)
st.subheader("Before vs after defenses")
if run.agent_mode:
    st.caption(
        "Seeds as written vs after the adaptive attacker rewrote the ones that failed "
        "(a seed counts as broken if any of its variants got through)."
    )
    stages = data.before_after_frame(run)
    stages["Short"] = stages["Configuration"].map(short_label)
    chart = (
        alt.Chart(stages)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Stage:N", title=None, sort=["Seeds as written", "After the agent"]),
            x=alt.X("ASR:Q", title="Attack success rate", axis=alt.Axis(format="%")),
            color=alt.Color("Stage:N", sort=["Seeds as written", "After the agent"], legend=None),
            row=alt.Row("Short:N", title=None, sort=None, header=alt.Header(labelAngle=0)),
            tooltip=[
                "Configuration:N",
                "Stage:N",
                alt.Tooltip("ASR:Q", format=".0%"),
                "Attacks through:N",
            ],
        )
        .properties(height=60)
    )
    st.altair_chart(chart, width="stretch")
    table = stages.drop(columns="Short")
elif run.benign:
    st.caption(
        "Left: attacks that got through (lower is better). Right: harmless questions a defense "
        "wrongly blocked (lower is better). Same rows, separate scales."
    )
    left, right = st.columns(2)
    left.altair_chart(
        bar_panel(summary, "ASR", "Attack success rate", "Attacks through"),
        width="stretch",
    )
    right.altair_chart(
        bar_panel(summary, "FRR", "False-refusal rate", "Harmless blocked"),
        width="stretch",
    )
    table = summary
else:
    st.altair_chart(
        bar_panel(summary, "ASR", "Attack success rate", "Attacks through"),
        width="stretch",
    )
    table = summary

with st.expander("Table view", icon=":material/table:"):
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Short": None,
            "ASR": st.column_config.ProgressColumn(
                "ASR", format="percent", min_value=0.0, max_value=1.0
            ),
            "FRR": st.column_config.ProgressColumn(
                "FRR", format="percent", min_value=0.0, max_value=1.0
            ),
            "Useful": st.column_config.NumberColumn("Useful answers", format="percent"),
        },
    )
    st.caption(
        "IF = input filter, CS = context sanitiser, OF = output leak filter, JG = judge guard"
    )

# Security vs usefulness: only when harmless questions were asked.
if run.benign and len(summary) > 1:
    st.subheader("Security vs usefulness")
    st.caption(
        "One dot per configuration. Down = fewer attacks get through, left = fewer harmless "
        "questions blocked. Hover a dot for its name; dots on the same spot scored the same."
    )
    points = summary.copy()
    named = {"none", "all"} | ({best.config} if best else set())
    points["Label"] = points["Configuration"].map(lambda c: short_label(c) if c in named else "")
    base = alt.Chart(points).encode(
        x=alt.X("FRR:Q", title="False-refusal rate", axis=alt.Axis(format="%", tickCount=5)),
        y=alt.Y("ASR:Q", title="Attack success rate", axis=alt.Axis(format="%", tickCount=5)),
        tooltip=[
            "Configuration:N",
            alt.Tooltip("ASR:Q", format=".0%"),
            alt.Tooltip("FRR:Q", format=".0%"),
            alt.Tooltip("Useful:Q", format=".0%", title="Useful answers"),
        ],
    )
    dots = base.mark_circle(size=110, opacity=0.85, stroke=ink["surface"], strokeWidth=1.5)
    names = base.mark_text(align="left", dx=8, dy=-6, fontSize=11, color=ink["ink"]).encode(
        text="Label:N"
    )
    st.altair_chart((dots + names).properties(height=320), width="stretch")

# ------------------------------------------------------------------- heatmap
st.subheader("Attack success by category")
st.caption(
    ("Brighter" if theme.mode() == "dark" else "Darker")
    + " = more attacks got through."
    + (" Agent runs count a seed as broken if any variant worked." if run.agent_mode else "")
)
cells = data.category_frame(run)
if not cells.empty:
    order = [short_label(c) for c in run.configs]
    base = alt.Chart(cells).encode(
        x=alt.X(
            "Category:N",
            title=None,
            sort=None,
            # "System prompt extraction" -> two lines instead of being cut off.
            axis=alt.Axis(
                labelAngle=0,
                orient="top",
                labelLimit=0,
                labelExpr=r"split(replace(datum.label, / (?=\S+$)/, '|'), '|')",
            ),
        ),
        y=alt.Y("Configuration:N", title=None, sort=order),
        tooltip=[
            alt.Tooltip("Full configuration:N", title="Configuration"),
            "Category:N",
            alt.Tooltip("ASR:Q", format=".0%"),
            "Attacks through:N",
        ],
    )
    heat = base.mark_rect(stroke=ink["surface"], strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("ASR:Q", scale=alt.Scale(domain=[0, 1]), legend=None)
    )
    # Dark cells get light text and vice versa; in dark mode the ramp runs the other way.
    strong = "#0b0b0b" if theme.mode() == "dark" else "#ffffff"
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("ASR:Q", format=".0%"),
        color=alt.condition(alt.datum.ASR > 0.5, alt.value(strong), alt.value(ink["ink"])),
    )
    st.altair_chart((heat + text).properties(height=max(140, 30 * len(order))), width="stretch")
    with st.expander("Table view", icon=":material/table:"):
        st.dataframe(
            cells.pivot(index="Full configuration", columns="Category", values="ASR").reindex(
                run.configs
            ),
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(name, format="percent")
                for name in cells["Category"].unique()
            },
        )

# ----------------------------------------------------------- judge agreement
agreement = common.agreement()
if agreement:
    from redrag.eval.judge_eval import render_agreement_markdown

    with st.expander("How reliable is the judge?", icon=":material/balance:"):
        st.markdown(render_agreement_markdown(agreement))

report = info.path / "report.html"
if report.exists():
    st.caption(f"Full report: `{common.display_path(report)}`")
