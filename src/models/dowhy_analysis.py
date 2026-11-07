"""DoWhy causal identification, estimation, and refutation analysis."""

from typing import Dict, List, Any

import pandas as pd


DEFAULT_CONFOUNDERS = [
    "Age",
    "PriorSpend",
    "VisitFrequency",
    "LoyaltyScore",
]

DEFAULT_REFUTERS = [
    "random_common_cause",
    "placebo_treatment_refuter",
    "data_subset_refuter",
]


def build_causal_graph(
    confounders: List[str],
    treatment: str,
    outcome: str,
) -> str:
    """Build a DAG where each confounder affects treatment and outcome."""

    edges = []

    for confounder in confounders:
        edges.append(f"{confounder} -> {treatment};")
        edges.append(f"{confounder} -> {outcome};")

    edges.append(f"{treatment} -> {outcome};")

    return "digraph {\n" + "\n".join(edges) + "\n}"


def run_dowhy_analysis(
    data: pd.DataFrame,
    treatment: str = "Discount",
    outcome: str = "Sales",
    confounders: List[str] = None,
    refutation_methods: List[str] = None,
) -> Dict[str, Any]:
    """
    Identify and estimate a causal effect using DoWhy,
    then run the requested DoWhy refutation tests.
    """

    from dowhy import CausalModel

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    confounders = (
        DEFAULT_CONFOUNDERS.copy()
        if confounders is None
        else list(confounders)
    )

    refutation_methods = (
        DEFAULT_REFUTERS.copy()
        if refutation_methods is None
        else list(refutation_methods)
    )

    if not confounders:
        raise ValueError("At least one confounder is required.")

    if len(set(confounders)) != len(confounders):
        raise ValueError("Confounders must not contain duplicates.")

    if treatment in confounders or outcome in confounders:
        raise ValueError(
            "Treatment and outcome cannot also be confounders."
        )

    required_columns = [treatment, outcome] + confounders
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    analysis_data = data[required_columns].copy()

    if analysis_data.isnull().any().any():
        raise ValueError(
            "Selected treatment, outcome, and confounder columns "
            "must not contain missing values."
        )

    treatment_values = set(analysis_data[treatment].unique())

    if treatment_values != {0, 1}:
        raise ValueError(
            "Treatment must be binary and contain both 0 and 1."
        )

    if not pd.api.types.is_numeric_dtype(analysis_data[outcome]):
        raise ValueError("Outcome must be numeric.")

    graph = build_causal_graph(
        confounders=confounders,
        treatment=treatment,
        outcome=outcome,
    )

    model = CausalModel(
        data=analysis_data,
        treatment=treatment,
        outcome=outcome,
        graph=graph,
    )

    estimand = model.identify_effect()

    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )

    refutations = {}

    for method_name in refutation_methods:
        refutations[method_name] = model.refute_estimate(
            estimand,
            estimate,
            method_name=method_name,
        )

    return {
        "graph": graph,
        "confounders": confounders,
        "treatment": treatment,
        "outcome": outcome,
        "estimand": estimand,
        "estimate": estimate,
        "ate": float(estimate.value),
        "refutations": refutations,
    }
