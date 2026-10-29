import pytest
import numpy as np
import pandas as pd

from src.data.synthetic import generate_synthetic_data
from src.models.dml_estimator import DoubleMLEstimator
from src.models.robustness import check_covariate_balance


FEATURES = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]


def test_dml_estimator_fit_predict():
    df = generate_synthetic_data(n_samples=300, random_state=42)

    dml = DoubleMLEstimator(
        n_splits=2,
        nuisance_type="linear",
        backend="native",
    )
    dml.fit(df[FEATURES], df["Discount"], df["Sales"])

    assert dml.is_fitted
    assert dml.ate_ is not None
    assert dml.ite_estimates_ is not None
    assert len(dml.ite_estimates_) == 300


def test_dml_predict_ite_handles_new_records():
    train_df = generate_synthetic_data(n_samples=300, random_state=42)
    new_df = generate_synthetic_data(n_samples=25, random_state=123)

    dml = DoubleMLEstimator(
        n_splits=2,
        nuisance_type="linear",
        backend="native",
    )
    dml.fit(
        train_df[FEATURES],
        train_df["Discount"],
        train_df["Sales"],
    )

    predictions = dml.predict_ite(new_df[FEATURES])

    assert len(predictions) == 25
    assert predictions.ndim == 1
    assert np.isfinite(predictions).all()


def test_covariate_balance():
    df = generate_synthetic_data(n_samples=200, random_state=42)
    features = ["Age", "PriorSpend"]

    bal = check_covariate_balance(
        df[features],
        df["Discount"],
        features,
    )

    assert len(bal) == 2
    assert "SMD" in bal.columns
    assert "Balanced" in bal.columns
    assert bal["SMD"].notna().all()


def test_covariate_balance_calculates_smd_correctly():
    X = pd.DataFrame({
        "FeatureA": [1.0, 2.0, 3.0, 4.0]
    })
    D = pd.Series([1, 1, 0, 0])

    result = check_covariate_balance(
        X,
        D,
        ["FeatureA"],
    )

    treated_mean = X.loc[D == 1, "FeatureA"].mean()
    control_mean = X.loc[D == 0, "FeatureA"].mean()
    treated_var = X.loc[D == 1, "FeatureA"].var()
    control_var = X.loc[D == 0, "FeatureA"].var()
    pooled_std = np.sqrt((treated_var + control_var) / 2.0) + 1e-6
    expected_smd = abs(treated_mean - control_mean) / pooled_std

    assert len(result) == 1
    assert result.iloc[0]["Feature"] == "FeatureA"
    assert result.iloc[0]["SMD"] == round(expected_smd, 4)
    assert bool(result.iloc[0]["Balanced"]) == (expected_smd < 0.10)