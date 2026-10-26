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
def test_engineer_features_calculates_expected_values():
    df = pd.DataFrame({
        "PriorSpend": [100.0, 120.0],
        "VisitFrequency": [10, 20],
        "Age": [20, 45],
        "DayOfWeek": [1, 6],
        "Sales": [0.0, 99.0],
        "Customers": [10, 100],
        "CompetitionDistance": [1000.0, 2000.0],
        "Month": [9, 10],
        "StateHoliday": [0, 1],
        "SchoolHoliday": [0, 0],
    })

    result = engineer_features(df)

    assert result["SpendPerVisit"].tolist() == pytest.approx([
        100.0 / (10 + 1e-5),
        120.0 / (20 + 1e-5),
    ])

    assert result["AgeGroup"].astype(str).tolist() == [
        "Young", "MiddleAged"
    ]

    assert result["IsWeekend"].tolist() == [0, 1]
    assert result["LogSales"].tolist() == pytest.approx([
        0.0, 4.605170186
    ])
    assert result["IsQ4"].tolist() == [0, 1]
    assert result["AnyHoliday"].tolist() == [0, 1]