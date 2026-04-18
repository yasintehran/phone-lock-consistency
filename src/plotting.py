from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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


def ensure_feature_figure_dir(base_dir: Path, feature_name: str) -> Path:
    feature_dir = base_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir


def prettify_label(name: str) -> str:
    label_map = {
        "baseline_change": "Baseline change",
        "variance_change": "Variance change",
        "zstandardized_dtw_distance": "Pattern-focused DTW distance",
        "magnitude_sensitive_dtw_distance": "Magnitude-sensitive DTW distance",
        "depression_change": "Depression change",
        "abs_depression_change": "Absolute depression change",
    }
    return label_map.get(name, name.replace("_", " ").capitalize())


def create_scatter_with_regression_plot(
    df,
    predictor: str,
    outcome: str,
    output_path: Path,
) -> None:
    plot_df = df.dropna(subset=[predictor, outcome]).copy()
    if plot_df.empty:
        return

    x = plot_df[predictor].astype(float).to_numpy()
    y = plot_df[outcome].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x, y, alpha=0.8)

    if len(plot_df) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = intercept + slope * xs
        ax.plot(xs, ys)

    ax.set_xlabel(prettify_label(predictor))
    ax.set_ylabel(prettify_label(outcome))

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_density_with_hdi(
    samples: np.ndarray,
    x_label: str,
    output_path: Path,
) -> None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        return

    mean_val = np.mean(samples)
    hdi_low, hdi_high = np.percentile(samples, [2.5, 97.5])

    kde = gaussian_kde(samples)
    x_grid = np.linspace(samples.min(), samples.max(), 500)
    y_grid = kde(x_grid)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x_grid, y_grid)
    ax.fill_between(x_grid, 0, y_grid, alpha=0.15)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(mean_val, linewidth=1.2)
    ax.hlines(
        y=y_grid.max() * 0.08,
        xmin=hdi_low,
        xmax=hdi_high,
        linewidth=3,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_posterior(trace, output_path: Path) -> None:
    if trace is None:
        return

    rho_samples = trace.posterior["rho"].values.flatten()
    _plot_density_with_hdi(
        samples=rho_samples,
        x_label="Posterior correlation (ρ)",
        output_path=output_path,
    )


def plot_regression_beta_posterior(trace, output_path: Path) -> None:
    if trace is None:
        return

    beta_samples = trace.posterior["beta"].values.flatten()
    _plot_density_with_hdi(
        samples=beta_samples,
        x_label="Posterior slope (β)",
        output_path=output_path,
    )


def plot_regression_parameter_panel(trace, output_path: Path) -> None:
    if trace is None:
        return

    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 10.0))

    for ax, samples, x_label in zip(
        axes,
        [alpha_samples, beta_samples, sigma_samples],
        ["Posterior intercept (α)", "Posterior slope (β)", "Posterior residual SD (σ)"],
    ):
        samples = np.asarray(samples, dtype=float)
        samples = samples[np.isfinite(samples)]

        if samples.size == 0:
            continue

        mean_val = np.mean(samples)
        hdi_low, hdi_high = np.percentile(samples, [2.5, 97.5])

        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min(), samples.max(), 500)
        y_grid = kde(x_grid)

        ax.plot(x_grid, y_grid)
        ax.fill_between(x_grid, 0, y_grid, alpha=0.15)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axvline(mean_val, linewidth=1.2)
        ax.hlines(
            y=y_grid.max() * 0.08,
            xmin=hdi_low,
            xmax=hdi_high,
            linewidth=3,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)