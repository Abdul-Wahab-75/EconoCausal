import pytest
import pandas as pd

from src.data.synthetic import generate_synthetic_data
from src.data.features import engineer_features, get_rossmann_feature_cols
from src.data.load_data import load_rossmann


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
    assert result["LogCustomers"].tolist() == pytest.approx([
        2.397895273, 4.615120517
    ])
    assert result["CustomersPerKmCompetition"].tolist() == pytest.approx([
        10.0 / 1.1,
        100.0 / 2.1,
    ])
    assert result["IsQ4"].tolist() == [0, 1]
    assert result["AnyHoliday"].tolist() == [0, 1]


def test_rossmann_feature_columns_exclude_target_and_identifiers():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "Date": ["2025-01-01", "2025-01-02"],
        "Sales": [100.0, 200.0],
        "Discount": [0, 1],
        "LogSales": [4.6, 5.3],
        "DayOfWeek": [1, 6],
        "Month": [1, 10],
        "Customers": [20, 50],
        "IsWeekend": [0, 1],
        "LogCustomers": [3.0, 3.9],
    })

    features = get_rossmann_feature_cols(df)

    assert "DayOfWeek" in features
    assert "Month" in features
    assert "Customers" in features
    assert "IsWeekend" in features
    assert "LogCustomers" in features

    assert "CustomerID" not in features
    assert "Date" not in features
    assert "Sales" not in features
    assert "Discount" not in features
    assert "LogSales" not in features
def test_load_rossmann_raises_when_train_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv not found"):
        load_rossmann(raw_dir=tmp_path)

def test_load_rossmann_filters_rows_and_merges_store_metadata(tmp_path):
    raw_dir = tmp_path / "rossmann"
    raw_dir.mkdir()

    train = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "DayOfWeek": [1, 2, 3, 4],
        "Date": [
            "2015-01-05",
            "2015-01-06",
            "2015-01-07",
            "2015-01-08",
        ],
        "Sales": [100.0, 0.0, 200.0, 300.0],
        "Customers": [20, 0, 30, 40],
        "Open": [1, 1, 0, 1],
        "Promo": [1, 0, 1, 0],
        "StateHoliday": ["0", "0", "a", "0"],
        "SchoolHoliday": [0, 0, 1, 0],
    })

    store = pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "c"],
        "CompetitionDistance": [1000.0, 2000.0],
        "Promo2": [0, 1],
    })

    train.to_csv(raw_dir / "train.csv", index=False)
    store.to_csv(raw_dir / "store.csv", index=False)

    result = load_rossmann(raw_dir=raw_dir)

    assert len(result) == 2
    assert result["CustomerID"].tolist() == [1, 2]
    assert result["Sales"].tolist() == [100.0, 300.0]
    assert result["Discount"].tolist() == [1, 0]

    assert result["StoreTypeCode"].tolist() == [1, 2]
    assert result["AssortmentCode"].tolist() == [1, 3]
    assert result["CompetitionDistance"].tolist() == [1000.0, 2000.0]
    assert result["Promo2"].tolist() == [0, 1]

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Year"].tolist() == [2015, 2015]
    assert result["Month"].tolist() == [1, 1]
    assert result["StateHoliday"].tolist() == [0, 0]
