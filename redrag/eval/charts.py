"""Charts for the HTML report, drawn with matplotlib and embedded as PNG images.

Design rules (consistent across every chart):
- One accent hue (blue) for data, gray for context, text in ink colours,
  never in the data colour. No rainbow: 16 defense combinations are told
  apart by *labels*, not by 16 colours nobody could distinguish.
- Magnitude uses one hue from light to dark (the heatmap).
- Never two y-scales on one chart. ASR and FRR get two side-by-side panels
  that share the same rows.
- Thin bars, light gridlines, no chart junk. Every chart in the report has a
  table next to it with the same numbers (the accessible view).

Each function returns a base64 PNG string, so the HTML report is one
self-contained file you can email or open offline.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # draw to images, never open a window
import matplotlib.pyplot as plt  # noqa: E402

# Palette (reference instance from the dataviz guidelines, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"  # series 1 (blue)
ACCENT_DARK = "#184f95"  # sequential step 600: the highlighted points
SEQUENTIAL = [  # blue, light -> dark (steps 100..700)
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip

FONT = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width: float, height: float):
    plt.rcParams["font.family"] = FONT
    figure = plt.figure(figsize=(width, height), dpi=144)
    figure.patch.set_facecolor(SURFACE)
    return figure


def _to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def short_label(config: str) -> str:
    """Compact name for a defense combination: "input_filter,judge_guard" -> "IF+JG"."""
    if config in ("none", "all"):
        return config
    initials = {
        "input_filter": "IF",
        "context_sanitiser": "CS",
        "output_leak_filter": "OF",
        "judge_guard": "JG",
    }
    return "+".join(initials.get(name, name) for name in config.split(","))


def tradeoff_chart(points: list[tuple[str, float, float]]) -> str:
    """Security vs usefulness: one dot per defense combination.

    `points` = [(config, asr, frr)]. Down = fewer attacks get through,
    left = fewer harmless questions blocked, so the best corner is bottom-left.
    """
    figure = _figure(7.2, 4.6)
    ax = figure.add_subplot()
    _style(ax)

    # Several combinations often score exactly the same (e.g. every combination
    # with judge_guard at 0% ASR). Draw one dot per position, with all their
    # names stacked in one label, instead of labels printed on top of each other.
    groups: dict[tuple[float, float], list[str]] = {}
    for config, asr, frr in points:
        groups.setdefault((round(frr * 100, 6), round(asr * 100, 6)), []).append(config)

    for (x, y), configs in groups.items():
        highlighted = any(config in ("none", "all") for config in configs)
        ax.scatter(
            x,
            y,
            s=90 if highlighted else 60,
            color=ACCENT_DARK if highlighted else ACCENT,
            edgecolors=SURFACE,  # thin surface ring keeps neighbouring dots apart
            linewidths=1.5,
            zorder=3,
        )
        names = [short_label(config) for config in configs]
        lines = [", ".join(names[i : i + 3]) for i in range(0, len(names), 3)]
        ax.annotate(
            "\n".join(lines),
            (x, y),
            xytext=(7, 3),
            textcoords="offset points",
            va="bottom",
            fontsize=8.5,
            linespacing=1.15,
            color=INK if highlighted else INK_SECONDARY,
            fontweight="bold" if highlighted else "normal",
        )
    ax.set_xlabel("False-refusal rate: harmless questions blocked (%)", color=INK_SECONDARY)
    ax.set_ylabel("Attack success rate (%)", color=INK_SECONDARY)
    top = max([asr * 100 for _, asr, _ in points] + [10])
    right = max([frr * 100 for _, _, frr in points] + [10])
    ax.set_ylim(-3, top * 1.15 + 4)
    ax.set_xlim(-1, right * 1.3 + 2)  # room for stacked labels on the right
    ax.text(
        0.99, 0.98, "better: down and to the left", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5, color=MUTED,
    )  # fmt: skip
    return _to_base64(figure)


def asr_frr_bars(rows: list[tuple[str, float | None, float | None]]) -> str:
    """Two panels sharing the same rows: ASR (left) and FRR (right), per combination."""
    labels = [short_label(config) for config, _, _ in rows]
    height = max(2.4, 0.32 * len(rows) + 1.2)
    figure = _figure(8.0, height)
    left, right = figure.subplots(1, 2, sharey=True)
    positions = list(range(len(rows)))
    for ax, index, title in ((left, 1, "Attack success rate"), (right, 2, "False-refusal rate")):
        _style(ax)
        ax.grid(axis="y", visible=False)
        values = [(row[index] or 0) * 100 for row in rows]
        ax.barh(positions, values, height=0.62, color=ACCENT, edgecolor=SURFACE, linewidth=1)
        axis_max = max(values + [10]) * 1.18
        gap = axis_max * 0.015  # label sits just past the bar end, whatever the scale
        for position, value, row in zip(positions, values, rows, strict=True):
            text = "n/a" if row[index] is None else f"{value:.0f}%"
            ax.text(value + gap, position, text, va="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, axis_max)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    left.set_yticks(positions, labels)
    left.invert_yaxis()  # first configuration ("none") at the top
    figure.tight_layout(w_pad=2.5)
    return _to_base64(figure)


def category_heatmap(
    categories: list[str], configs: list[str], values: list[list[float | None]]
) -> str:
    """ASR for each (defense combination, category): rows = combinations, columns = categories."""
    figure = _figure(max(5.5, 1.25 * len(categories) + 2.2), max(2.4, 0.36 * len(configs) + 1.3))
    ax = figure.add_subplot()
    ax.set_facecolor(SURFACE)
    for row, config_values in enumerate(values):
        for column, value in enumerate(config_values):
            if value is None:
                colour, text, ink = "#f0efec", "n/a", MUTED
            else:
                step = min(int(value * (len(SEQUENTIAL) - 1) + 0.5), len(SEQUENTIAL) - 1)
                colour = SEQUENTIAL[step]
                text = f"{value:.0%}"
                ink = "#ffffff" if step >= 6 else INK
            ax.add_patch(
                plt.Rectangle((column, row), 1, 1, facecolor=colour, edgecolor=SURFACE, linewidth=2)
            )
            ax.text(
                column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=8.5, color=ink
            )
    ax.set_xlim(0, len(categories))
    ax.set_ylim(len(configs), 0)
    ax.set_xticks([i + 0.5 for i in range(len(categories))], categories, fontsize=8.5)
    ax.set_yticks(
        [i + 0.5 for i in range(len(configs))], [short_label(c) for c in configs], fontsize=8.5
    )
    ax.tick_params(length=0, labelcolor=INK_SECONDARY)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _to_base64(figure)
