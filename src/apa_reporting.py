from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import PipelineConfig
from .plotting import create_scatter_with_regression_plot, ensure_feature_figure_dir


def apa_sentence_correlation(row: pd.Series) -> str:
    weeks_text = f"{int(row['number_of_weeks'])}-week"
    if pd.isna(row["corr_mean"]):
        return (
            f"For {row['feature']} in the {weeks_text} window, the Bayesian correlation "
            f"could not be estimated because there were too few usable observations."
        )
    return (
        f"For {row['feature']} in the {weeks_text} window, a Bayesian correlation analysis "
        f"of {row['predictor']} and {row['outcome']} yielded "
        f"r = {row['corr_mean']:.3f}, 95% HDI [{row['corr_hdi_low']:.3f}, {row['corr_hdi_high']:.3f}], "
        f"BF10 = {row['corr_bf10']:.3f} (n = {int(row['sample_size'])})."
    )


def apa_sentence_regression(row: pd.Series) -> str:
    weeks_text = f"{int(row['number_of_weeks'])}-week"
    if pd.isna(row["reg_beta_mean"]):
        return (
            f"For {row['feature']} in the {weeks_text} window, the Bayesian regression "
            f"could not be estimated because there were too few usable observations."
        )
    return (
        f"For {row['feature']} in the {weeks_text} window, a Bayesian regression analysis "
        f"predicting {row['outcome']} from {row['predictor']} yielded "
        f"beta = {row['reg_beta_mean']:.3f}, 95% HDI [{row['reg_beta_hdi_low']:.3f}, {row['reg_beta_hdi_high']:.3f}], "
        f"BF10 = {row['reg_bf10']:.3f} (n = {int(row['sample_size'])})."
    )


def save_apa_outputs(
    metric_tables: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    summary_path = config.tables_dir / "bayesian_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    corr_sentences = [apa_sentence_correlation(row) for _, row in summary_df.iterrows()]
    reg_sentences = [apa_sentence_regression(row) for _, row in summary_df.iterrows()]

    (config.text_dir / "apa_correlation_sentences.txt").write_text(
        "\n".join(corr_sentences),
        encoding="utf-8",
    )
    (config.text_dir / "apa_regression_sentences.txt").write_text(
        "\n".join(reg_sentences),
        encoding="utf-8",
    )

    predictor_to_outcome = {
        "baseline_change": "depression_change",
        "variance_change": "depression_change",
        "zstandardized_dtw_distance": "abs_depression_change",
        "magnitude_sensitive_dtw_distance": "abs_depression_change",
    }

    predictor_short_names = {
        "baseline_change": "baseline_change",
        "variance_change": "variance_change",
        "zstandardized_dtw_distance": "pattern_dtw",
        "magnitude_sensitive_dtw_distance": "magnitude_dtw",
    }

    for _, row in summary_df.iterrows():
        feature_name = row["feature"]
        weeks = int(row["number_of_weeks"])
        predictor = row["predictor"]
        outcome = predictor_to_outcome[predictor]

        metric_df = metric_tables[feature_name]
        plot_df = metric_df[metric_df["number_of_weeks"] == weeks]

        scatter_feature_dir = ensure_feature_figure_dir(config.scatter_figure_dir, feature_name)
        short_predictor = predictor_short_names[predictor]

        output_path = (
            scatter_feature_dir
            / f"{feature_name}_{weeks}w_{short_predictor}_scatter.png"
        )

        create_scatter_with_regression_plot(
            df=plot_df,
            predictor=predictor,
            outcome=outcome,
            output_path=output_path,
        )