import pandas as pd

from src.apa_reporting import save_apa_outputs
from src.bayesian_analysis import run_bayesian_analysis_for_metric_df
from src.config import PipelineConfig
from src.utils import ensure_directories, save_json


def main() -> None:
    config = PipelineConfig()
    ensure_directories(config)

    metric_tables = {}
    all_summaries = []
    for feature_name in ["total_number", "total_duration", "avg_duration"]:
        metric_df = pd.read_csv(config.analysis_ready_dir / f"mean_var_pattern_data_{feature_name}.csv")
        metric_tables[feature_name] = metric_df
        feature_results, feature_summary = run_bayesian_analysis_for_metric_df(metric_df, feature_name, config)
        all_summaries.append(feature_summary)
        save_json(feature_results, config.models_dir / f"{feature_name}_bayesian_results.json")
        feature_summary.to_csv(config.tables_dir / f"{feature_name}_bayesian_summary.csv", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    save_apa_outputs(metric_tables, summary_df, config)


if __name__ == "__main__":
    main()
