from src.config import PipelineConfig
from src.pipeline import run_pipeline


def main() -> None:
    config = PipelineConfig(
        phone_lock_dir="data/raw/phonelock",
        phq9_csv_path="data/raw/phq9/PHQ-9.csv",
        output_root="outputs",
        data_root="data",
        random_seed=42,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
