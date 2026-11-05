from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _calculate_smd(
    X: pd.DataFrame,
    treatment: pd.Series,
) -> pd.DataFrame:
    """Calculate absolute standardized mean differences."""
    treated = X.loc[treatment == 1]
    control = X.loc[treatment == 0]

    records = []

    for column in X.columns:
        treated_mean = treated[column].mean()
        control_mean = control[column].mean()
        treated_var = treated[column].var(ddof=1)
        control_var = control[column].var(ddof=1)

        pooled_std = np.sqrt((treated_var + control_var) / 2.0)

        if pooled_std == 0 or not np.isfinite(pooled_std):
            smd = 0.0 if treated_mean == control_mean else np.inf
        else:
            smd = abs(treated_mean - control_mean) / pooled_std

        records.append({
            "Feature": column,
            "Treated_Mean": float(treated_mean),
            "Control_Mean": float(control_mean),
            "SMD": float(smd),
            "Balanced": bool(smd < 0.10),
        })

    return pd.DataFrame(records)


def propensity_score_matching(
    X: pd.DataFrame,
    treatment: pd.Series,
    caliper: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform 1:1 nearest-neighbour propensity score matching.

    Matching uses estimated propensity scores on the logit scale,
    common support, and no replacement of control observations.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(treatment, pd.Series):
        raise TypeError("treatment must be a pandas Series.")

    if not X.index.equals(treatment.index):
        raise ValueError("X and treatment must have identical indexes.")

    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X must contain only finite numeric values.")

    if treatment.isna().any() or set(treatment.unique()) != {0, 1}:
        raise ValueError("Treatment must contain both binary values 0 and 1.")

    if not isinstance(caliper, (int, float)) or not np.isfinite(caliper) or caliper <= 0:
        raise ValueError("caliper must be a positive finite number.")

    X_numeric = X.astype(float)
    D = treatment.astype(int)

    if D.value_counts().min() < 2:
        raise ValueError("At least two observations per treatment group are required.")

    # Estimate propensity scores using observed covariates only.
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
    )
    propensity_model.fit(X_numeric, D)

    propensity = propensity_model.predict_proba(X_numeric)[:, 1]
    propensity = np.clip(propensity, 1e-6, 1 - 1e-6)

    # Restrict matching to the overlapping propensity-score region.
    treated_mask = D.to_numpy() == 1
    control_mask = D.to_numpy() == 0

    lower = max(
        propensity[treated_mask].min(),
        propensity[control_mask].min(),
    )
    upper = min(
        propensity[treated_mask].max(),
        propensity[control_mask].max(),
    )

    if lower >= upper:
        raise ValueError("No common propensity-score support exists.")

    in_support = (propensity >= lower) & (propensity <= upper)

    treated_positions = np.flatnonzero(treated_mask & in_support)
    control_positions = np.flatnonzero(control_mask & in_support)

    if len(treated_positions) == 0 or len(control_positions) == 0:
        raise ValueError("No treated-control observations remain in common support.")

    # Match on logit propensity score using a standard caliper.
    logit_scores = logit(propensity)
    caliper_width = caliper * np.std(
        logit_scores[in_support],
        ddof=1,
    )

    if not np.isfinite(caliper_width) or caliper_width <= 0:
        raise ValueError("Unable to calculate a valid matching caliper.")

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(logit_scores[control_positions].reshape(-1, 1))

    distances, indices = neighbors.kneighbors(
        logit_scores[treated_positions].reshape(-1, 1)
    )

    # Greedy matching without replacement.
    candidate_order = np.argsort(logit_scores[treated_positions])
    available_controls = set(range(len(control_positions)))
    pairs = []

    for candidate in candidate_order:
        control_neighbor = int(indices[candidate, 0])
        distance = float(distances[candidate, 0])

        if control_neighbor not in available_controls:
            remaining = sorted(available_controls)

            if not remaining:
                break

            available_scores = logit_scores[
                control_positions[np.asarray(remaining)]
            ].reshape(-1, 1)

            nearest = NearestNeighbors(n_neighbors=1)
            nearest.fit(available_scores)

            new_distance, new_index = nearest.kneighbors(
                [[logit_scores[treated_positions[candidate]]]]
            )

            control_neighbor = remaining[int(new_index[0, 0])]
            distance = float(new_distance[0, 0])

        if distance > caliper_width:
            continue

        available_controls.remove(control_neighbor)

        treated_position = int(treated_positions[candidate])
        control_position = int(control_positions[control_neighbor])

        pairs.append({
            "treated_index": X.index[treated_position],
            "control_index": X.index[control_position],
            "treated_propensity": float(propensity[treated_position]),
            "control_propensity": float(propensity[control_position]),
            "logit_distance": distance,
        })

    if not pairs:
        raise ValueError(
            "No matches found within the caliper. Consider reviewing overlap "
            "or selecting a justified caliper."
        )

    matched_pairs = pd.DataFrame(pairs)

    treated_indices = matched_pairs["treated_index"].tolist()
    control_indices = matched_pairs["control_index"].tolist()

    matched_X = pd.concat([
        X_numeric.loc[treated_indices],
        X_numeric.loc[control_indices],
    ])

    matched_D = pd.Series(
        [1] * len(treated_indices) + [0] * len(control_indices),
        index=matched_X.index,
        name="treatment",
    )

    propensity_scores = pd.Series(
        propensity,
        index=X.index,
        name="propensity_score",
    )

    return {
        "propensity_scores": propensity_scores,
        "matched_pairs": matched_pairs,
        "matched_treated": X.loc[treated_indices].copy(),
        "matched_control": X.loc[control_indices].copy(),
        "balance_before": _calculate_smd(X_numeric, D),
        "balance_after": _calculate_smd(matched_X, matched_D),
        "n_treated_before": int(treated_mask.sum()),
        "n_control_before": int(control_mask.sum()),
        "n_treated_matched": len(treated_indices),
        "n_control_matched": len(control_indices),
        "caliper_width": float(caliper_width),
        "common_support": (float(lower), float(upper)),
    }
