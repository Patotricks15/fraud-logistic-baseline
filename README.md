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

Real data, full dataset (284,807 rows, 492 frauds). Splits: train 199,368 rows (384 frauds), validation 42,718 (56), test 42,721 (52).

| Window | PR-AUC | ROC-AUC | Recall @ 1% alerts | Brier |
|---|---|---|---|---|
| Validation (C = 1.0) | 0.819 | 0.981 | 0.911 | n/a |
| Test (tuned C = 1.0) | 0.690 | 0.976 | 0.808 | 0.023 |

Validation PR-AUC by C: 0.01 → 0.792, 0.1 → 0.819, 1.0 → 0.819, 10.0 → 0.819. The tuned value is C = 1.0. Test confusion matrix at threshold 0.5: TN 41,546, FP 1,123, FN 8, TP 44.

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
