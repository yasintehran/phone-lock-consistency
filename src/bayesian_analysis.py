from __future__ import annotations

from typing import List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.stats import gaussian_kde

from .clean_analysis_sample import apply_analysis_specific_exclusion
from .config import PipelineConfig
from .utils import evidence_label_from_bf10


def calculate_correlation_bayes_factor(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    minimum_points_for_analysis: int,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < minimum_points_for_analysis:
        return None, None

    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)
    observed_data = np.vstack([x_std, y_std]).T

    with pm.Model() as model:
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
        cov = pm.math.stack([
            pm.math.stack([1.0, rho]),
            pm.math.stack([rho, 1.0]),
        ])
        pm.MvNormal("likelihood", mu=np.zeros(2), cov=cov, observed=observed_data)
        trace = pm.sample(
            draws=n_samples,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    rho_samples = trace.posterior["rho"].values.flatten()
    try:
        kde = gaussian_kde(rho_samples)
        posterior_at_0 = float(kde(0)[0])
        prior_at_0 = 1 / (0.99 - (-0.99))
        bf01 = posterior_at_0 / prior_at_0
        bf10 = 1 / bf01 if bf01 != 0 else np.inf
    except Exception:
        bf10 = None
    return bf10, trace


def calculate_regression_bayes_factor(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    minimum_points_for_analysis: int,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < minimum_points_for_analysis:
        return None, None

    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return None, None

    x_z = (x - np.mean(x)) / x_std
    y_z = (y - np.mean(y)) / y_std

    try:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("beta", mu=0, sigma=5)
            sigma = pm.HalfNormal("sigma", sigma=5)
            mu = alpha + beta * x_z
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_z)
            trace = pm.sample(
                draws=n_samples,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=False,
            )

        beta_posterior = trace.posterior["beta"].values.flatten()
        kde = stats.gaussian_kde(beta_posterior)
        posterior_density_at_zero = float(kde(0)[0])
        prior_density_at_zero = float(stats.norm(0, 5).pdf(0))
        bf01 = posterior_density_at_zero / prior_density_at_zero
        bf10 = 1 / bf01 if bf01 != 0 else np.inf
        return bf10, trace
    except Exception:
        return None, None


def summarize_correlation(trace, bayes_factor):
    if trace is None:
        return None
    summary = az.summary(trace, var_names=["rho"], hdi_prob=0.95)
    row = summary.loc["rho"]
    return {
        "mean": float(row["mean"]),
        "sd": float(row["sd"]),
        "hdi_2_5%": float(row.get("hdi_2.5%", row.get("hdi_3%", np.nan))),
        "hdi_97_5%": float(row.get("hdi_97.5%", row.get("hdi_97%", np.nan))),
        "bayes_factor": None if bayes_factor is None else float(bayes_factor),
        "interpretation": evidence_label_from_bf10(bayes_factor),
    }


def summarize_regression(trace, bayes_factor):
    if trace is None:
        return None
    summary = az.summary(trace, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95)

    def pull(name):
        row = summary.loc[name]
        return {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_2_5%": float(row.get("hdi_2.5%", row.get("hdi_3%", np.nan))),
            "hdi_97_5%": float(row.get("hdi_97.5%", row.get("hdi_97%", np.nan))),
        }

    return {
        "intercept": pull("alpha"),
        "slope": pull("beta"),
        "sigma": pull("sigma"),
        "bayes_factor": None if bayes_factor is None else float(bayes_factor),
        "interpretation": evidence_label_from_bf10(bayes_factor),
    }


def run_bayesian_analysis_for_metric_df(
    metric_df: pd.DataFrame,
    feature_name: str,
    config: PipelineConfig,
) -> tuple[dict, pd.DataFrame]:
    predictors = [
        "baseline_change",
        "variance_change",
        "zstandardized_dtw_distance",
        "magnitude_sensitive_dtw_distance",
    ]
    outcomes = {
        "baseline_change": "depression_change",
        "variance_change": "depression_change",
        "zstandardized_dtw_distance": "abs_depression_change",
        "magnitude_sensitive_dtw_distance": "abs_depression_change",
    }

    all_results: dict = {}
    summary_rows: List[dict] = []

    for weeks, df_week in metric_df.groupby("number_of_weeks"):
        week_results: dict = {}
        for predictor in predictors:
            outcome = outcomes[predictor]
            filtered_df = apply_analysis_specific_exclusion(df_week, predictor, outcome)
            x = filtered_df[predictor].to_numpy(dtype=float)
            y = filtered_df[outcome].to_numpy(dtype=float)

            corr_bf10, corr_trace = calculate_correlation_bayes_factor(
                x=x,
                y=y,
                n_samples=config.n_samples_correlation,
                tune=config.tune,
                chains=config.chains,
                cores=config.cores,
                random_seed=config.random_seed,
                minimum_points_for_analysis=config.minimum_points_for_analysis,
            )
            reg_bf10, reg_trace = calculate_regression_bayes_factor(
                x=x,
                y=y,
                n_samples=config.n_samples_regression,
                tune=config.tune,
                chains=config.chains,
                cores=config.cores,
                random_seed=config.random_seed,
                minimum_points_for_analysis=config.minimum_points_for_analysis,
            )

            corr_summary = summarize_correlation(corr_trace, corr_bf10)
            reg_summary = summarize_regression(reg_trace, reg_bf10)
            week_results[predictor] = {
                "feature": feature_name,
                "number_of_weeks": int(weeks),
                "predictor": predictor,
                "outcome": outcome,
                "sample_size": int(len(filtered_df)),
                "correlation": corr_summary,
                "regression": reg_summary,
            }

            summary_rows.append(
                {
                    "feature": feature_name,
                    "number_of_weeks": int(weeks),
                    "predictor": predictor,
                    "outcome": outcome,
                    "sample_size": int(len(filtered_df)),
                    "corr_mean": None if corr_summary is None else corr_summary["mean"],
                    "corr_hdi_low": None if corr_summary is None else corr_summary["hdi_2_5%"],
                    "corr_hdi_high": None if corr_summary is None else corr_summary["hdi_97_5%"],
                    "corr_bf10": None if corr_summary is None else corr_summary["bayes_factor"],
                    "corr_interpretation": None if corr_summary is None else corr_summary["interpretation"],
                    "reg_beta_mean": None if reg_summary is None else reg_summary["slope"]["mean"],
                    "reg_beta_hdi_low": None if reg_summary is None else reg_summary["slope"]["hdi_2_5%"],
                    "reg_beta_hdi_high": None if reg_summary is None else reg_summary["slope"]["hdi_97_5%"],
                    "reg_bf10": None if reg_summary is None else reg_summary["bayes_factor"],
                    "reg_interpretation": None if reg_summary is None else reg_summary["interpretation"],
                }
            )

        all_results[f"{feature_name}_{weeks}_weeks"] = week_results

    summary_df = pd.DataFrame(summary_rows)
    return all_results, summary_df
