from __future__ import annotations

from typing import Dict

import pandas as pd


def summarize_daily_features(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    data_summary: Dict[str, pd.DataFrame] = {}
    for participant_id, df in data_dict.items():
        df_summary = (
            df.groupby("date")
            .agg(
                total_number=("date", "size"),
                total_duration=("duration", "sum"),
            )
            .reset_index()
        )
        df_summary["avg_duration"] = df_summary["total_duration"] / df_summary["total_number"]
        data_summary[participant_id] = df_summary
    return data_summary


def build_feature_matrices(data_summary: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    total_number_df = pd.DataFrame()
    total_duration_df = pd.DataFrame()
    avg_duration_df = pd.DataFrame()

    all_dates = pd.to_datetime([])
    for df_summary in data_summary.values():
        all_dates = all_dates.append(pd.DatetimeIndex(df_summary["date"]))
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(all_dates.unique())))

    for participant_id, df_summary in data_summary.items():
        participant_df = df_summary.copy()
        participant_df["date"] = pd.to_datetime(participant_df["date"])
        participant_df = participant_df.set_index("date")

        total_number_df[participant_id] = participant_df["total_number"].reindex(all_dates)
        total_duration_df[participant_id] = participant_df["total_duration"].reindex(all_dates)
        avg_duration_df[participant_id] = participant_df["avg_duration"].reindex(all_dates)

    total_number_df.index.name = "date"
    total_duration_df.index.name = "date"
    avg_duration_df.index.name = "date"

    return {
        "total_number": total_number_df,
        "total_duration": total_duration_df,
        "avg_duration": avg_duration_df,
    }
