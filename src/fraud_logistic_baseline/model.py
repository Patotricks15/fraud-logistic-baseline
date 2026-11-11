"""Logistic regression baseline with standardized inputs and balanced class weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import AMOUNT, PCA_FEATURES

RANDOM_STATE = 42
FEATURES = [*PCA_FEATURES, "log_amount"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out[AMOUNT])
    return out


def build_logistic(C: float = 1.0) -> Pipeline:
    """Scaler and logistic regression fitted on training data only."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def coefficient_table(model: Pipeline, feature_names: list[str] = FEATURES) -> pd.DataFrame:
    """Standardized coefficients, sorted by absolute size."""
    coefs = model.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
