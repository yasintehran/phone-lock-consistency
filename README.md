# Phone Lock Consistency Pipeline

A reproducible Python analysis pipeline for evaluating the temporal consistency of smartphone phone-unlock behavioral features in relation to changes in depression scores.

This repository was developed as part of a Master's thesis in Psychology at Uppsala University. The project uses the StudentLife dataset to examine whether behavioral features derived from smartphone unlock logs show consistency relative to changes in depressive symptoms over time.

---

## Project Purpose

Digital phenotyping studies often prioritize prediction accuracy, but less attention has been given to whether behavioral features behave consistently in relation to the psychological construct they are intended to reflect.

This project evaluates whether phone-unlock features demonstrate temporal consistency with depression change, using:

* baseline level change
* variability change
* temporal pattern similarity
* magnitude-sensitive temporal similarity

The goal is to assess whether consistency evaluation may be useful as a methodological step before or after predictive modeling.

---

## Data Source

This project uses data derived from the StudentLife dataset.

The raw dataset is **not included** in this public repository. Please obtain the dataset from the official source and place the required files in the folder structure shown below.

---

## Repository Structure

```text
phone-lock-consistency/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── analysis_ready/
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── text/
│
├── src/
│   ├── config.py
│   ├── load_phone_data.py
│   ├── load_phq_data.py
│   ├── feature_extraction.py
│   ├── save_features.py
│   ├── consistency_metrics.py
│   ├── clean_analysis_sample.py
│   ├── bayesian_analysis.py
│   ├── plotting.py
│   ├── apa_reporting.py
│   └── pipeline.py
│
└── scripts/
    ├── run_preprocessing.py
    ├── run_consistency_metrics.py
    ├── run_bayesian_analysis.py
    └── run_all.py
```

---

## Main Behavioral Features

Daily features extracted from phone unlock logs:

1. Total number of unlocks
2. Total phone usage duration
3. Average duration per unlock

---

## Consistency Metrics

For 1–4 week individualized pre/post windows, the pipeline computes:

* baseline pre/post values and change
* variance pre/post values and change
* pattern-focused DTW distance (z-standardized time series)
* magnitude-sensitive DTW distance (raw values)
* depression score change
* absolute depression score change

---

## Statistical Analysis

Bayesian analyses are implemented using PyMC.

Outputs include:

* Bayesian correlations
* Bayesian regressions
* Bayes factors
* posterior summaries
* APA-style text outputs
* figures and summary tables

---

## Reproducibility

Random seeds are fixed for:

* NumPy
* Python random
* PyMC sampling

This improves reproducibility of model estimates.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

From the project root:

```bash
python -m scripts.run_all
```

---

## Notes

* Raw data are not included.
* Generated outputs are excluded from the public repository.
* Folder placeholders are included to preserve the workflow structure.

---

## Thesis Context

This repository accompanies a Master's thesis focused on temporal consistency as a methodological consideration in digital phenotyping research.

