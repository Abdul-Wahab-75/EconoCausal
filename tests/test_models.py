import pytest
import numpy as np
import pandas as pd
from src.data.synthetic import generate_synthetic_data
from src.models.dml_estimator import DoubleMLEstimator
from src.models.robustness import check_covariate_balance

def test_dml_estimator_fit_predict():
    df = generate_synthetic_data(n_samples=300, random_state=42)
    features = ["Age", "PriorSpend", "VisitFrequency", "LoyaltyScore"]
    
    dml = DoubleMLEstimator(n_splits=2, nuisance_type="linear", backend="native")
    dml.fit(df[features], df["Discount"], df["Sales"])
    
    assert dml.is_fitted
    assert dml.ate_ is not None
    assert dml.ite_estimates_ is not None
    assert len(dml.ite_estimates_) == 300


def test_covariate_balance():
    df = generate_synthetic_data(n_samples=200, random_state=42)
    features = ["Age", "PriorSpend"]

    bal = check_covariate_balance(
        df[features],
        df["Discount"],
        features
    )

    assert len(bal) == 2
    assert "SMD" in bal.columns
    assert "Balanced" in bal.columns
    assert bal["SMD"].notna().all()
