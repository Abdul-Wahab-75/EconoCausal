from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10
) -> float:
    """Calculate Population Stability Index using reference quantile bins."""

    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Both datasets must contain valid numeric observations.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        return 0.0 if np.all(current == reference[0]) else float("inf")

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-6
    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.20,
    significance_level: float = 0.05,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compare reference and current numerical feature distributions.

    A feature is flagged when PSI reaches the configured threshold
    or the two-sample KS test p-value is below the significance level.
    """

    if reference.empty or current.empty:
        raise ValueError("Reference and current datasets cannot be empty.")

    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1.")

    if psi_threshold < 0:
        raise ValueError("psi_threshold cannot be negative.")

    common_columns = [
        col for col in reference.columns
        if col in current.columns
        and pd.api.types.is_numeric_dtype(reference[col])
        and pd.api.types.is_numeric_dtype(current[col])
    ]

    if not common_columns:
        raise ValueError("No common numerical features found in both datasets.")

    results = []

    for feature in common_columns:
        ref_values = pd.to_numeric(
            reference[feature], errors="coerce"
        ).dropna()

        cur_values = pd.to_numeric(
            current[feature], errors="coerce"
        ).dropna()

        if len(ref_values) < 2 or len(cur_values) < 2:
            continue

        psi = calculate_psi(ref_values, cur_values, n_bins=n_bins)
        ks_result = ks_2samp(ref_values, cur_values)

        psi_drift = psi >= psi_threshold
        ks_drift = float(ks_result.pvalue) < significance_level

        results.append({
            "feature": feature,
            "reference_count": int(len(ref_values)),
            "current_count": int(len(cur_values)),
            "psi": psi,
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "psi_drift": bool(psi_drift),
            "ks_drift": bool(ks_drift),
            "drift_detected": bool(psi_drift or ks_drift),
        })

    if not results:
        raise ValueError("No numerical features have enough valid observations.")

    drifted_features = [
        item["feature"] for item in results
        if item["drift_detected"]
    ]

    return {
        "status": "success",
        "reference_observations": len(reference),
        "current_observations": len(current),
        "features_checked": len(results),
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "feature_results": results,
        "thresholds": {
            "psi_threshold": psi_threshold,
            "significance_level": significance_level,
        },
    }
