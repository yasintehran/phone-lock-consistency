from __future__ import annotations

import pandas as pd

from .config import PHQ_RESPONSE_MAP


def load_phq9_data(phq_csv_path: str) -> pd.DataFrame:
    phq = pd.read_csv(phq_csv_path)
    scored = phq.copy()
    scored.replace(PHQ_RESPONSE_MAP, inplace=True)

    scored["id_type"] = scored["uid"].astype(str) + "_" + scored["type"].astype(str)
    scored = scored.set_index("id_type")

    columns_to_drop = [col for col in ["uid", "type", "Response"] if col in scored.columns]
    scored = scored.drop(columns=columns_to_drop)
    scored["total_score"] = scored.sum(axis=1, numeric_only=True)
    scored = scored.reset_index()

    scored["participant_id"] = scored["id_type"].str.split("_").str[0]
    scored["timepoint"] = scored["id_type"].str.split("_").str[1]

    score_table = scored.pivot_table(
        index="participant_id",
        columns="timepoint",
        values="total_score",
        aggfunc="first",
    ).reset_index()

    rename_map = {}
    for col in score_table.columns:
        lower = str(col).lower()
        if lower == "pre":
            rename_map[col] = "phq_pre"
        elif lower == "post":
            rename_map[col] = "phq_post"
    score_table = score_table.rename(columns=rename_map)

    if "phq_pre" not in score_table.columns:
        score_table["phq_pre"] = pd.NA
    if "phq_post" not in score_table.columns:
        score_table["phq_post"] = pd.NA

    score_table["depression_change"] = score_table["phq_post"] - score_table["phq_pre"]
    score_table["abs_depression_change"] = score_table["depression_change"].abs()
    score_table["has_valid_depression_change"] = (
        score_table["phq_pre"].notna() & score_table["phq_post"].notna()
    )
    return score_table
