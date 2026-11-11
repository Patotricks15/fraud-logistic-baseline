# %% [markdown]
# # FD01: Logistic regression baseline
# Source for the FD01 notebook. Convert with `jupytext --to ipynb` and run on Kaggle, or set `FRAUD_DATA_PATH` locally.
# Dataset: Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`), time-ordered split 70/15/15.
# Train on the earliest 70% of time, tune on the next 15%, report the last 15% once.

# %%
from pathlib import Path

import pandas as pd

from fraud_logistic_baseline.data import load_creditcard, time_split
from fraud_logistic_baseline.metrics import brier, confusion_at_threshold, metric_summary, pr_auc
from fraud_logistic_baseline.model import FEATURES, add_features, build_logistic, coefficient_table
from fraud_logistic_baseline import plots

FIG_DIR = Path("../reports/figures") if Path.cwd().name == "notebooks" else Path("reports/figures")
ALERT_RATE = 0.01
C_GRID = [0.01, 0.1, 1.0, 10.0]

# %%
df = add_features(load_creditcard())
split = time_split(df)
X, y = df[FEATURES], df["Class"]
print(f"rows: {len(df):,}  frauds: {int(y.sum())}")
print(f"train end (Time): {split.train_end:,.0f}s  val end: {split.val_end:,.0f}s")
for name, mask in (("train", split.train), ("val", split.val), ("test", split.test)):
    print(f"{name}: {int(mask.sum()):,} rows, {int(y[mask].sum())} frauds")

# %% [markdown]
# ## Baseline (C = 1.0) on validation

# %%
baseline = build_logistic(C=1.0).fit(X[split.train], y[split.train])
val_scores = baseline.predict_proba(X[split.val])[:, 1]
print(metric_summary(y[split.val], val_scores, ALERT_RATE))

# %% [markdown]
# ## Tune C on validation PR-AUC only

# %%
val_rows = []
for C in C_GRID:
    model = build_logistic(C=C).fit(X[split.train], y[split.train])
    scores = model.predict_proba(X[split.val])[:, 1]
    val_rows.append({"C": C, "val_pr_auc": pr_auc(y[split.val], scores)})
tuning = pd.DataFrame(val_rows)
print(tuning)
best_C = float(tuning.sort_values("val_pr_auc", ascending=False).iloc[0]["C"])
print(f"chosen C: {best_C}")

# %% [markdown]
# ## Final model on the test window (reported once)

# %%
model = build_logistic(C=best_C).fit(X[split.train], y[split.train])
test_probs = model.predict_proba(X[split.test])[:, 1]
test_y = y[split.test]
title_suffix = f"creditcardfraud, time test split, logistic C={best_C}"

summary = metric_summary(test_y, test_probs, ALERT_RATE)
summary["brier"] = brier(test_y, test_probs)
print(summary)
print(confusion_at_threshold(test_y, test_probs, threshold=0.5))

# %%
plots.plot_pr_curve(test_y, test_probs, f"PR curve, {title_suffix}", FIG_DIR / "fd01_pr_curve.png")
plots.plot_calibration(
    test_y, test_probs, f"Reliability diagram, {title_suffix}", FIG_DIR / "fd01_calibration.png"
)
coefs = coefficient_table(model)
plots.plot_coefficients(coefs, f"Standardized coefficients, {title_suffix}", FIG_DIR / "fd01_coefficients.png")
print(coefs.head(10))
