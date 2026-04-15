from src.config import PipelineConfig
from src.feature_extraction import build_feature_matrices, summarize_daily_features
from src.load_phone_data import load_phone_lock_data
from src.load_phq_data import load_phq9_data
from src.save_features import save_feature_files
from src.utils import ensure_directories


def main() -> None:
    config = PipelineConfig()
    ensure_directories(config)
    phone_data = load_phone_lock_data(config.phone_lock_dir)
    _ = load_phq9_data(config.phq9_csv_path)
    daily_summaries = summarize_daily_features(phone_data)
    feature_matrices = build_feature_matrices(daily_summaries)
    save_feature_files(feature_matrices, config.processed_data_dir)


if __name__ == "__main__":
    main()
