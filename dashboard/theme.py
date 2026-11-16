"""Colours and small styling helpers for the dashboard.

Most of the look comes from `.streamlit/config.toml` (light and dark themes,
chart palettes). This module holds what the config can't express: colours for
things we draw ourselves (heatmap labels, the lineage tree), formatting
helpers, and a little CSS. Restyling the dashboard means editing these two files.

Colour rules (same as the HTML report):
- one blue accent for data; magnitude = one blue ramp, light to dark;
- a verdict is always written in words, never shown by colour alone;
- text uses ink colours, never the data colour.
"""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "accent": "#2a78d6",
        "verdicts": {  # lineage boxes
            "success": "#184f95",
            "unsure": "#cde2fb",
            "fail": "#e1e0d9",
            "error": "#f0efec",
        },
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "accent": "#3987e5",
        "verdicts": {
            "success": "#3987e5",
            "unsure": "#1c5cab",
            "fail": "#383835",
            "error": "#2c2c2a",
        },
    },
}

VERDICT_ICONS = {
    "success": ":material/gpp_bad:",  # the attack got through
    "fail": ":material/verified_user:",  # the defense held
    "unsure": ":material/help:",
    "error": ":material/error:",
}

CSS = """
<style>
/* Tighter page, readable line length for prose. */
.block-container { padding-top: 2.2rem; max-width: 1200px; }
div[data-testid="stMetricLabel"] p { font-size: 0.85rem; }
/* Chat bubbles in transcripts: keep long attack text wrapping. */
div[data-testid="stChatMessage"] p { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
"""


def mode() -> str:
    """ "light" or "dark": what the viewer is using right now."""
    try:
        kind = st.context.theme.type
    except Exception:  # older Streamlit, or outside a running app (tests)
        kind = None
    return kind if kind in PALETTE else "light"


def colours() -> dict:
    return PALETTE[mode()]


def apply() -> None:
    """Call once per page run."""
    st.markdown(CSS, unsafe_allow_html=True)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def verdict_text(verdict: str) -> str:
    """e.g. ":material/gpp_bad: success" - icon plus the word, never colour alone."""
    return f"{VERDICT_ICONS.get(verdict, '')} {verdict}"


def escape_md(text: str) -> str:
    """Show model output literally: no Markdown formatting, links or HTML from it."""
    special = "\\`*_{}[]<>()#+-.!|~$"
    return "".join("\\" + char if char in special else char for char in text)
