from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from dtaidistance import dtw as dtaidistance_dtw

from .config import PipelineConfig


def ensure_directories(config: PipelineConfig) -> None:
    for path in [
        config.raw_data_dir,
        config.processed_data_dir,
        config.analysis_ready_dir,
        config.output_root_path,
        config.tables_dir,
        config.figures_dir,
        config.models_dir,
        config.text_dir,
        config.scatter_figure_dir,
        config.correlation_posterior_dir,
        config.regression_beta_posterior_dir,
        config.regression_parameters_posterior_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_participant_id(filename: str) -> Optional[str]:
    match = re.search(r"u\d+", filename)
    return match.group() if match else None


def safe_std(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    std = np.std(arr)
    return std if std != 0 else np.nan


def dtw_distance(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float(dtaidistance_dtw.distance(x, y))


def evidence_label_from_bf10(bf10: Optional[float]) -> str:
    if bf10 is None or (isinstance(bf10, float) and (np.isnan(bf10) or np.isinf(bf10))):
        return "Bayes factor could not be calculated"
    if bf10 < 1:
        bf01 = 1 / bf10
        if bf01 < 3:
            return "Anecdotal evidence for the null"
        if bf01 < 10:
            return "Moderate evidence for the null"
        if bf01 < 30:
            return "Strong evidence for the null"
        if bf01 < 100:
            return "Very strong evidence for the null"
        return "Extreme evidence for the null"
    if bf10 < 3:
        return "Anecdotal evidence for the alternative"
    if bf10 < 10:
        return "Moderate evidence for the alternative"
    if bf10 < 30:
        return "Strong evidence for the alternative"
    if bf10 < 100:
        return "Very strong evidence for the alternative"
    return "Extreme evidence for the alternative"
