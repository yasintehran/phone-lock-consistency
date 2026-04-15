# Phone Lock Consistency

This repository contains a modular Python pipeline for reproducing the phone-lock consistency analyses.

## Repository layout

- `data/raw/`: raw phone-lock and PHQ-9 input files
- `data/processed/`: processed intermediate data such as feature files
- `data/analysis_ready/`: analysis-ready metric tables
- `outputs/tables/`: summary tables and CSV outputs
- `outputs/figures/`: thesis-ready figures
- `outputs/models/`: JSON summaries of Bayesian results
- `outputs/text/`: APA-style result sentences
- `src/`: core analysis modules
- `scripts/`: runnable entry points

## Expected raw inputs

- Phone lock CSV files in a folder such as `data/raw/phonelock/`
- Raw PHQ-9 CSV file such as `data/raw/phq9/PHQ-9.csv`

## Reproducibility

Random seeding is set in the configuration and applied to both NumPy and Python's `random` module, and also passed into PyMC sampling.

## Run the full pipeline

```bash
python scripts/run_all.py
```

## Notes

- DTW uses `dtaidistance.dtw.distance` only. The code fails loudly if the dependency is missing.
- The two DTW metrics are named to reflect what they compute:
  - `zstandardized_dtw_distance`
  - `magnitude_sensitive_dtw_distance`
