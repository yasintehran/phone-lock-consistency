from src.clean_analysis_sample import apply_global_exclusion
from src.config import PipelineConfig
from src.consistency_metrics import get_mean_var_pattern_data, save_metric_files
from src.load_phq_data import load_phq9_data
from src.utils import ensure_directories
import pandas as pd


def main() -> None:
    config = PipelineConfig()
    ensure_directories(config)
    phq_df = load_phq9_data(config.phq9_csv_path)

    metric_tables = {}
    for feature_name in ["total_number", "total_duration", "avg_duration"]:
        feature_df = pd.read_csv(config.processed_data_dir / f"{feature_name}.csv", index_col="date")
        feature_df.index = pd.to_datetime(feature_df.index)
        metric_df = get_mean_var_pattern_data(feature_df, phq_df, config.week_windows)
        metric_tables[feature_name] = apply_global_exclusion(metric_df)

    save_metric_files(metric_tables, config.analysis_ready_dir)


if __name__ == "__main__":
    main()
