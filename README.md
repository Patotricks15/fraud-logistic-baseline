# fraud-logistic-baseline (FD01)

**Question:** What does a transparent, explainable baseline achieve on card fraud detection?

## Data

- Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013, 284,807 rows, 492 frauds).
- License per Kaggle API: Database Open License. Read date: 2026-11-08. The raw file is read in place and is not committed. See [data/README.md](data/README.md).
- `Time` covers about two days, so the time split spans two days only.

## Method

- Features: `V1`-`V28` (PCA components, used as provided) and `log(1 + Amount)`. `Time` is excluded from features so that the model does not learn the time position.
- Model: `StandardScaler` + `LogisticRegression(class_weight="balanced", random_state=42)`.
- Baseline: `C = 1.0`. Tuned: `C` chosen from `{0.01, 0.1, 1, 10}` by validation PR-AUC only.
- Split (time-ordered, no shuffle): earliest 70% of `Time` for training, next 15% for validation, last 15% for test. The test window is scored once.
- Metrics: PR-AUC (primary), ROC-AUC, recall at a 1% alert rate, confusion matrix at threshold 0.5, Brier score and reliability diagram.

## Results

Not yet produced. The notebook needs the Kaggle dataset, which is not downloaded to this workstation. After running the notebook on Kaggle, fill in this table:

| Window | PR-AUC | ROC-AUC | Recall @ 1% alerts | Brier |
|---|---|---|---|---|
| Validation (C = 1.0) | pending | pending | pending | pending |
| Test (tuned C) | pending | pending | pending | pending |

## Decision

A logistic model with class weights is the reference point for FD02. Any tree model must beat its test PR-AUC to justify its extra complexity.

## Limitation

Features are anonymized PCA components, so the model cannot be checked against business rules. The two-day window is too short to show seasonal or monthly drift.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
# On Kaggle: attach mlg-ulb/creditcardfraud, upload the notebook, Run All.
# Locally (development only): export FRAUD_DATA_PATH=/path/to/creditcard.csv
make notebook
```

Figures are written to `reports/figures/`: `fd01_pr_curve.png`, `fd01_calibration.png`, `fd01_coefficients.png`.

## Layout

```text
src/fraud_logistic_baseline/  data loading, time split, metrics, model, plots
notebooks/                    jupytext source (.py) and paired notebook (.ipynb)
tests/                        split, metrics and scaler-leakage tests
reports/figures/              exported figures
```
