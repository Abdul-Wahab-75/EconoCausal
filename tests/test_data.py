import pytest
import pandas as pd
from src.data.synthetic import generate_synthetic_data
from src.data.features import engineer_features

def test_synthetic_data_generation():
    df = generate_synthetic_data(n_samples=500, random_state=42)
    assert len(df) == 500
    assert "Discount" in df.columns
    assert "Sales" in df.columns
    assert "TrueITE" in df.columns
    assert df["Discount"].isin([0, 1]).all()
    assert (df["Sales"] >= 0).all()

def test_feature_engineering():
    df = generate_synthetic_data(n_samples=100, random_state=42)
    df_feat = engineer_features(df)
    assert "SpendPerVisit" in df_feat.columns
    assert "AgeGroup" in df_feat.columns
