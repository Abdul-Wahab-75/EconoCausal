import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_uplift_curve,
    compute_qini_score,
    summarize_causal_metrics,
)


def test_uplift_and_qini_computation():
    y = np.array([10, 20, 15, 30, 25, 40, 5, 12, 18, 22])
    t = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    ite = np.array([5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1])

    curve_df = compute_uplift_curve(y, t, ite, n_bins=2)

    assert len(curve_df) == 2
    assert "CumulativeQini" in curve_df.columns

    score = compute_qini_score(curve_df)

    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_summarize_causal_metrics():
    ite_preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_ite = np.array([1.1, 1.9, 3.1, 3.9, 5.2])

    summary = summarize_causal_metrics(ite_preds, true_ite)

    assert "Mean_Estimated_ITE" in summary
    assert "GroundTruth_RMSE" in summary
    assert summary["Mean_Estimated_ITE"] == 3.0

    # Verify ground-truth error metrics against expected values.
    assert summary["GroundTruth_RMSE"] == pytest.approx(np.sqrt(0.016))
    assert summary["GroundTruth_MAE"] == pytest.approx(0.12)
