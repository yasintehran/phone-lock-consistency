from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .apa_reporting import save_apa_outputs
from .bayesian_analysis import run_bayesian_analysis_for_metric_df
from .clean_analysis_sample import apply_global_exclusion
from .config import PipelineConfig
from .consistency_metrics import get_mean_var_pattern_data, save_metric_files
from .feature_extraction import build_feature_matrices, summarize_daily_features
from .load_phone_data import load_phone_lock_data
from .load_phq_data import load_phq9_data
from .plotting import set_apa_plot_style
from .save_features import save_feature_files
from .utils import ensure_directories, save_json


def run_pipeline(config: PipelineConfig) -> dict:
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    ensure_directories(config)
    set_apa_plot_style()

    phone_data = load_phone_lock_data(config.phone_lock_dir)
    phq_df = load_phq9_data(config.phq9_csv_path)

    daily_summaries = summarize_daily_features(phone_data)
    feature_matrices = build_feature_matrices(daily_summaries)
    save_feature_files(feature_matrices, config.processed_data_dir)

    metric_tables = {}
    globally_filtered_metric_tables = {}

    for feature_name, feature_df in feature_matrices.items():
        metric_df = get_mean_var_pattern_data(
            feature_df=feature_df,
            phq_df=phq_df,
            week_windows=config.week_windows,
        )
        metric_tables[feature_name] = metric_df
        globally_filtered_metric_tables[feature_name] = apply_global_exclusion(metric_df)

    save_metric_files(metric_tables, config.analysis_ready_dir)

    all_results = {}
    all_summaries = []
    for feature_name, metric_df in globally_filtered_metric_tables.items():
        feature_results, feature_summary = run_bayesian_analysis_for_metric_df(
            metric_df=metric_df,
            feature_name=feature_name,
            config=config,
        )
        all_results[feature_name] = feature_results
        all_summaries.append(feature_summary)

        save_json(feature_results, config.models_dir / f"{feature_name}_bayesian_results.json")
        feature_summary.to_csv(config.tables_dir / f"{feature_name}_bayesian_summary.csv", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    if not summary_df.empty:
        save_apa_outputs(metric_tables=globally_filtered_metric_tables, summary_df=summary_df, config=config)

    save_json({"config": config.as_dict()}, config.output_root_path / "run_config.json")

    return {
        "phq_df": phq_df,
        "feature_matrices": feature_matrices,
        "metric_tables": metric_tables,
        "globally_filtered_metric_tables": globally_filtered_metric_tables,
        "bayesian_summary_df": summary_df,
        "all_results": all_results,
    }
