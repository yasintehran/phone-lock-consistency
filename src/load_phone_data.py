from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import extract_participant_id


def load_phone_lock_data(phone_lock_dir: str) -> Dict[str, pd.DataFrame]:
    phone_lock_path = Path(phone_lock_dir)
    data_dict: Dict[str, pd.DataFrame] = {}

    for file_path in sorted(phone_lock_path.glob("*.csv")):
        participant_id = extract_participant_id(file_path.name)
        if participant_id is None:
            continue

        df = pd.read_csv(file_path)
        required_columns = {"start", "end"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} is missing required columns: {missing}")

        df = df.copy()
        df["duration"] = df["end"] - df["start"]
        df["datetime"] = pd.to_datetime(df["start"], unit="s")
        df["date"] = df["datetime"].dt.date
        data_dict[participant_id] = df

    return data_dict
