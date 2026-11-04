from typing import Any, Dict
import math

import pandas as pd


REQUIRED_RESULT_KEYS = {
    "n_targeted",
    "total_cost",
    "budget_limit",
    "total_incremental_value",
    "estimated_roi",
}

REQUIRED_POLICY_COLUMNS = {"Policy", "Incremental_Gain"}

EXPECTED_POLICIES = {
    "Causal DML Optimal",
    "High-Spender Heuristic",
    "Random Targeting",
}


def _finite_number(value: Any, field_name: str) -> float:
    """Convert a value to a finite float or raise a clear error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def create_strategy_summary(
    knapsack_result: Dict[str, Any],
    policy_comparison: pd.DataFrame,
    dataset_name: str,
    currency_symbol: str = "₹",
) -> Dict[str, Any]:
    """
    Create a business-friendly strategy summary from actual optimizer
    and allocation-policy comparison results.

    All financial values are estimates produced by the project model.
    """
    missing_keys = REQUIRED_RESULT_KEYS - set(knapsack_result)
    if missing_keys:
        raise ValueError(
            f"Knapsack result is missing keys: {sorted(missing_keys)}"
        )

    if not isinstance(policy_comparison, pd.DataFrame):
        raise TypeError("policy_comparison must be a pandas DataFrame.")

    missing_columns = REQUIRED_POLICY_COLUMNS - set(policy_comparison.columns)
    if missing_columns:
        raise ValueError(
            f"Policy comparison is missing columns: {sorted(missing_columns)}"
        )

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")

    if not isinstance(currency_symbol, str) or not currency_symbol:
        raise ValueError("currency_symbol must be a non-empty string.")

    budget = _finite_number(knapsack_result["budget_limit"], "budget_limit")
    spent = _finite_number(knapsack_result["total_cost"], "total_cost")
    incremental_value = _finite_number(
        knapsack_result["total_incremental_value"],
        "total_incremental_value",
    )
    roi = _finite_number(knapsack_result["estimated_roi"], "estimated_roi")
    targeted_value = _finite_number(
        knapsack_result["n_targeted"], "n_targeted"
    )

    if budget < 0 or spent < 0 or targeted_value < 0:
        raise ValueError("Budget, spend, and targeted count cannot be negative.")

    if not targeted_value.is_integer():
        raise ValueError("n_targeted must be a whole number.")

    if spent > budget + 1e-6:
        raise ValueError("Optimizer spend cannot exceed the budget limit.")

    available_policies = set(policy_comparison["Policy"].dropna())
    missing_policies = EXPECTED_POLICIES - available_policies
    if missing_policies:
        raise ValueError(
            f"Policy comparison is missing policies: {sorted(missing_policies)}"
        )

    policy_gains = {}
    for policy_name in EXPECTED_POLICIES:
        matching_rows = policy_comparison[
            policy_comparison["Policy"] == policy_name
        ]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one row for policy: {policy_name}"
            )

        policy_gains[policy_name] = _finite_number(
            matching_rows.iloc[0]["Incremental_Gain"],
            f"{policy_name} Incremental_Gain",
        )

    utilization = (spent / budget * 100.0) if budget > 0 else 0.0
    remaining_budget = max(budget - spent, 0.0)

    causal_gain = policy_gains["Causal DML Optimal"]
    heuristic_gain = policy_gains["High-Spender Heuristic"]
    random_gain = policy_gains["Random Targeting"]

    if heuristic_gain > 0:
        comparison_pct = (
            (causal_gain - heuristic_gain) / heuristic_gain
        ) * 100.0
        if comparison_pct >= 0:
            comparison_text = (
                f"Causal DML estimates {comparison_pct:.1f}% more "
                "incremental value than the High-Spender Heuristic."
            )
        else:
            comparison_text = (
                f"Causal DML estimates {abs(comparison_pct):.1f}% less "
                "incremental value than the High-Spender Heuristic."
            )
    else:
        comparison_text = (
            "The High-Spender Heuristic has zero or negative estimated "
            "incremental value, so a percentage comparison is not meaningful."
        )

    best_policy = max(policy_gains, key=policy_gains.get)

    overview = (
        f"Using the {dataset_name} dataset, the Causal DML optimizer "
        f"selected {int(targeted_value):,} records. It allocated "
        f"{currency_symbol}{spent:,.2f} from a budget of "
        f"{currency_symbol}{budget:,.2f}, leaving "
        f"{currency_symbol}{remaining_budget:,.2f} unspent. "
        f"The model estimates an incremental value of "
        f"{currency_symbol}{incremental_value:,.2f} and an ROI of {roi:.2f}x."
    )

    recommendation = (
        f"Among the three evaluated allocation policies, {best_policy} "
        f"has the highest estimated incremental value "
        f"({currency_symbol}{policy_gains[best_policy]:,.2f}). "
        "Review the policy comparison before making a campaign decision."
    )

    caveat = (
        "These figures are model-based estimates from observational data. "
        "They are not guaranteed realized revenue or profit. Validate the "
        "recommended strategy through a controlled experiment before "
        "large-scale deployment."
    )

    return {
        "dataset_name": dataset_name,
        "targeted_count": int(targeted_value),
        "budget": budget,
        "budget_spent": spent,
        "budget_remaining": remaining_budget,
        "budget_utilization_pct": utilization,
        "estimated_incremental_value": incremental_value,
        "estimated_roi": roi,
        "policy_gains": policy_gains,
        "best_policy": best_policy,
        "comparison_text": comparison_text,
        "overview": overview,
        "recommendation": recommendation,
        "caveat": caveat,
    }
