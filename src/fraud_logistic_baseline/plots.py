"""Figures for the FD01 report. Each title names the dataset, split and model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pr_curve(y_true, scores, title: str, path: str | Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_calibration(y_true, probs, title: str, path: str | Path, bins: int = 10) -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_coefficients(table, title: str, path: str | Path, top: int = 15) -> Path:
    top_table = table.head(top).iloc[::-1]
    colors = np.where(top_table["coefficient"] >= 0, "tab:red", "tab:blue")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top_table["feature"], top_table["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title(title)
    return save(fig, path)
