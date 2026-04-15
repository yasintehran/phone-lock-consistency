from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .utils import dtw_distance, safe_std


def get_mean_var_pattern_data(
    feature_df: pd.DataFrame,
    phq_df: pd.DataFrame,
    week_windows: Iterable[int],
) -> pd.DataFrame:
    phq_lookup = phq_df.set_index("participant_id")
    person_ids = feature_df.columns.tolist()
    statistic_data: List[dict] = []

    for number_of_weeks in week_windows:
        for person in person_ids:
            person_data = feature_df[person].dropna()
            if len(person_data) == 0:
                continue

            person_start = person_data.index.min()
            person_end = person_data.index.max()

            person_first_weeks = feature_df[person][
                (feature_df.index >= person_start)
                & (feature_df.index < person_start + pd.Timedelta(days=7 * number_of_weeks))
            ]
            person_last_weeks = feature_df[person][
                (feature_df.index <= person_end)
                & (feature_df.index > person_end - pd.Timedelta(days=7 * number_of_weeks))
            ]

            baseline_pre = np.nanmean(person_first_weeks)
            baseline_post = np.nanmean(person_last_weeks)
            baseline_change = baseline_post - baseline_pre

            variance_pre = np.nanvar(person_first_weeks)
            variance_post = np.nanvar(person_last_weeks)
            variance_change = variance_post - variance_pre

            filtered_first_weeks = person_first_weeks[~np.isnan(person_first_weeks)]
            filtered_last_weeks = person_last_weeks[~np.isnan(person_last_weeks)]

            magnitude_sensitive_dtw_distance = dtw_distance(filtered_first_weeks, filtered_last_weeks)

            std_pre = safe_std(filtered_first_weeks)
            std_post = safe_std(filtered_last_weeks)
            if np.isnan(std_pre) or np.isnan(std_post):
                zstandardized_dtw_distance = np.nan
            else:
                first_weeks_z = (filtered_first_weeks - baseline_pre) / std_pre
                last_weeks_z = (filtered_last_weeks - baseline_post) / std_post
                zstandardized_dtw_distance = dtw_distance(first_weeks_z, last_weeks_z)

            if person in phq_lookup.index and bool(phq_lookup.loc[person, "has_valid_depression_change"]):
                depression_change = phq_lookup.loc[person, "depression_change"]
                abs_depression_change = phq_lookup.loc[person, "abs_depression_change"]
            else:
                depression_change = np.nan
                abs_depression_change = np.nan

            statistic_data.append(
                {
                    "participant_id": person,
                    "number_of_weeks": number_of_weeks,
                    "baseline_pre": baseline_pre,
                    "baseline_post": baseline_post,
                    "baseline_change": baseline_change,
                    "variance_pre": variance_pre,
                    "variance_post": variance_post,
                    "variance_change": variance_change,
                    "zstandardized_dtw_distance": zstandardized_dtw_distance,
                    "magnitude_sensitive_dtw_distance": magnitude_sensitive_dtw_distance,
                    "depression_change": depression_change,
                    "abs_depression_change": abs_depression_change,
                }
            )

    return pd.DataFrame(statistic_data)


def save_metric_files(metric_tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_name, df in metric_tables.items():
        df.to_csv(output_dir / f"mean_var_pattern_data_{feature_name}.csv", index=False)
