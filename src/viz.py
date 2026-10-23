"""Estilo visual compartilhado (notebook de EDA + graficos de avaliacao).

Paleta categorica daltonico-segura: azul = cliente retido, laranja = churn.
Heatmaps usam rampa sequencial azul (magnitude) ou divergente azul<->vermelho
com ponto medio cinza (polaridade).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#6250d6"
RED = "#e34948"

SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
CHURN_PALETTE = {0: BLUE, 1: ORANGE}
CHURN_LABELS = {0: "Retido", 1: "Churn"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#2a78d6", "#9ec5f4", "#f0efec", "#f2a39e", "#e34948"]
)


def apply_style() -> None:
    """Aplica o tema global do matplotlib (limpo, grade discreta, sem molduras)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    )


def pct_axis(ax, axis: str = "y") -> None:
    """Formata um eixo (0-1) como porcentagem."""
    from matplotlib.ticker import PercentFormatter

    fmt = PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def annotate_bars(ax, fmt: str = "{:.0%}", pad: float = 3) -> None:
    """Rotula o topo de cada barra (rotulo direto, em tinta secundaria)."""
    for container in ax.containers:
        labels = [fmt.format(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=pad, color=INK_SECONDARY, fontsize=9)


def save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
