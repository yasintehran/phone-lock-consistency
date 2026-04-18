from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple


PHQ_RESPONSE_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


@dataclass
class PipelineConfig:
    phone_lock_dir: str = "data/raw/phonelock"
    phq9_csv_path: str = "data/raw/phq9/PHQ-9.csv"
    data_root: str = "data"
    output_root: str = "outputs"
    week_windows: Tuple[int, ...] = (1, 2, 3, 4)
    n_samples_correlation: int = 5000
    n_samples_regression: int = 2000
    tune: int = 1000
    chains: int = 2
    cores: int = 1
    random_seed: int = 42
    minimum_points_for_analysis: int = 3

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def raw_data_dir(self) -> Path:
        return self.data_root_path / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_root_path / "processed"

    @property
    def analysis_ready_dir(self) -> Path:
        return self.data_root_path / "analysis_ready"

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)

    @property
    def tables_dir(self) -> Path:
        return self.output_root_path / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root_path / "figures"

    @property
    def models_dir(self) -> Path:
        return self.output_root_path / "models"

    @property
    def text_dir(self) -> Path:
        return self.output_root_path / "text"

    @property
    def scatter_figure_dir(self) -> Path:
        return self.figures_dir / "scatter_regression"

    @property
    def correlation_posterior_dir(self) -> Path:
        return self.figures_dir / "correlation_posterior"

    @property
    def regression_beta_posterior_dir(self) -> Path:
        return self.figures_dir / "regression_beta_posterior"

    @property
    def regression_parameters_posterior_dir(self) -> Path:
        return self.figures_dir / "regression_parameters_posterior" 
    
    
    def as_dict(self) -> dict:
        return asdict(self)
