import numpy as np
import pandas as pd

from fraud_logistic_baseline.model import FEATURES, add_features, build_logistic


def _data(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.uniform(1, 500, n)
    df["Class"] = (rng.uniform(size=n) < 0.05).astype(int)
    return add_features(df)


def test_scaler_is_fit_on_training_rows_only():
    df = _data()
    train, test = df.iloc[:300], df.iloc[300:]
    model = build_logistic().fit(train[FEATURES], train["Class"])
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, train[FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, df[FEATURES].mean().to_numpy())
    assert len(test) == 100


def test_log_amount_is_added_without_changing_input():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Amount": rng.uniform(1, 500, 10)})
    out = add_features(df)
    assert "log_amount" not in df.columns
    np.testing.assert_allclose(out["log_amount"], np.log1p(df["Amount"]))
