from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def save_feature_files(feature_matrices: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_name, df in feature_matrices.items():
        df.to_csv(output_dir / f"{feature_name}.csv")
