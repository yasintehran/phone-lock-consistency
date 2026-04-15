from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_apa_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.6,
    })


def create_scatter_with_regression_plot(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = df.dropna(subset=[predictor, outcome]).copy()
    if plot_df.empty:
        return

    x = plot_df[predictor].astype(float)
    y = plot_df[outcome].astype(float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x, y, alpha=0.8)

    if len(plot_df) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = intercept + slope * xs
        ax.plot(xs, ys)

    ax.set_xlabel(predictor.replace("_", " ").title())
    ax.set_ylabel(outcome.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
