"""Markdown + matplotlib reporting helpers (headless 'Agg' backend, consistent palette)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def grouped_bars(ax: plt.Axes, categories: Sequence[str], series: dict[str, Sequence[float]], ylim: tuple[float, float] | None = (0, 1.22),
                 value_labels: bool = True) -> None:
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(categories))
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92, label=name, color=SERIES[i % len(SERIES)],
                      edgecolor=SURFACE, linewidth=1.0)
        if value_labels:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=TEXT2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    if n > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=min(n, 4), fontsize=8)
