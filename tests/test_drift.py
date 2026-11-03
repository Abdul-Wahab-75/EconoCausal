import numpy as np
import pandas as pd
import pytest

from src.evaluation.drift import calculate_psi, detect_data_drift


def test_same_data_has_no_drift():
    rng = np.random.RandomState(42)
    reference = pd.DataFrame({
        "Age": rng.normal(35, 8, 500),
        "PriorSpend": rng.normal(2500, 500, 500),
    })

    result = detect_data_drift(reference, reference.copy())

    assert result["status"] == "success"
    assert result["drift_detected"] is False
    assert result["features_checked"] == 2
    assert result["drifted_features"] == []


def test_shifted_data_detects_drift():
    rng = np.random.RandomState(42)
    reference = pd.DataFrame({
        "Age": rng.normal(35, 8, 500),
        "PriorSpend": rng.normal(2500, 500, 500),
    })

    current = reference.copy()
    current["Age"] += 15
    current["PriorSpend"] += 1500

    result = detect_data_drift(reference, current)

    assert result["drift_detected"] is True
    assert "Age" in result["drifted_features"]
    assert "PriorSpend" in result["drifted_features"]


def test_unchanged_feature_is_not_flagged():
    rng = np.random.RandomState(42)
    reference = pd.DataFrame({
        "Age": rng.normal(35, 8, 500),
        "VisitFrequency": rng.normal(5, 2, 500),
    })

    current = reference.copy()
    current["Age"] += 15

    result = detect_data_drift(reference, current)

    visit_result = next(
        item for item in result["feature_results"]
        if item["feature"] == "VisitFrequency"
    )

    assert visit_result["drift_detected"] is False


def test_calculate_psi_returns_finite_value():
    reference = pd.Series(np.arange(100, dtype=float))
    current = pd.Series(np.arange(100, dtype=float))

    result = calculate_psi(reference, current)

    assert np.isfinite(result)
    assert result == pytest.approx(0.0)


def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        detect_data_drift(pd.DataFrame(), pd.DataFrame())


def test_invalid_significance_level_raises_error():
    data = pd.DataFrame({"Age": [20, 25, 30, 35]})

    with pytest.raises(ValueError, match="significance_level"):
        detect_data_drift(data, data, significance_level=1.5)
