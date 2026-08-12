"""Shared matplotlib style: the dataviz skill's validated default palette (light mode).

Categorical slots are assigned in fixed order, never cycled past slot 5 (more series
than that folds into small multiples). Sequential encoding uses the single blue hue
ramp. Grid/axes recessive; marks thin; text in ink tokens, never series colors.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
"""Fixed categorical order: blue, orange, aqua, yellow, magenta."""

BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("posthoc_blues", BLUE_RAMP)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 150,
        }
    )


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")
