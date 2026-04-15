from __future__ import annotations

import pandas as pd


def apply_global_exclusion(metric_df: pd.DataFrame) -> pd.DataFrame:
    df = metric_df.copy()
    df["depression_change"] = pd.to_numeric(df["depression_change"], errors="coerce")
    df["abs_depression_change"] = pd.to_numeric(df["abs_depression_change"], errors="coerce")
    return df[df["depression_change"].notna()].copy()


def apply_analysis_specific_exclusion(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> pd.DataFrame:
    return df.dropna(subset=[predictor, outcome]).copy()
