import pandas as pd
import pytest

from src.dashboard.strategy_summary import create_strategy_summary


@pytest.fixture
def knapsack_result():
    return {
        "n_targeted": 10,
        "total_cost": 800.0,
        "budget_limit": 1000.0,
        "total_incremental_value": 4000.0,
        "estimated_roi": 5.0,
    }


@pytest.fixture
def policy_comparison():
    return pd.DataFrame([
        {"Policy": "Causal DML Optimal", "Incremental_Gain": 4000.0},
        {"Policy": "High-Spender Heuristic", "Incremental_Gain": 3000.0},
        {"Policy": "Random Targeting", "Incremental_Gain": 2500.0},
    ])


def test_strategy_summary_calculates_budget_and_roi(
    knapsack_result, policy_comparison
):
    summary = create_strategy_summary(
        knapsack_result,
        policy_comparison,
        dataset_name="Synthetic Benchmark",
        currency_symbol="₹",
    )

    assert summary["targeted_count"] == 10
    assert summary["budget_spent"] == pytest.approx(800.0)
    assert summary["budget_remaining"] == pytest.approx(200.0)
    assert summary["budget_utilization_pct"] == pytest.approx(80.0)
    assert summary["estimated_incremental_value"] == pytest.approx(4000.0)
    assert summary["estimated_roi"] == pytest.approx(5.0)
    assert summary["best_policy"] == "Causal DML Optimal"
    assert "Synthetic Benchmark" in summary["overview"]
    assert "not guaranteed realized revenue or profit" in summary["caveat"]


def test_policy_comparison_is_independent_of_row_order(
    knapsack_result, policy_comparison
):
    shuffled = policy_comparison.iloc[::-1].reset_index(drop=True)

    summary = create_strategy_summary(
        knapsack_result,
        shuffled,
        dataset_name="Test Dataset",
    )

    assert summary["policy_gains"]["Causal DML Optimal"] == pytest.approx(4000.0)
    assert summary["policy_gains"]["High-Spender Heuristic"] == pytest.approx(3000.0)
    assert summary["policy_gains"]["Random Targeting"] == pytest.approx(2500.0)
    assert summary["best_policy"] == "Causal DML Optimal"


def test_strategy_summary_rejects_spend_above_budget(
    knapsack_result, policy_comparison
):
    knapsack_result["total_cost"] = 1200.0

    with pytest.raises(ValueError, match="cannot exceed"):
        create_strategy_summary(
            knapsack_result,
            policy_comparison,
            dataset_name="Test Dataset",
        )


def test_strategy_summary_rejects_missing_policy(
    knapsack_result, policy_comparison
):
    incomplete = policy_comparison[
        policy_comparison["Policy"] != "Random Targeting"
    ]

    with pytest.raises(ValueError, match="missing policies"):
        create_strategy_summary(
            knapsack_result,
            incomplete,
            dataset_name="Test Dataset",
        )


def test_strategy_summary_rejects_non_finite_values(
    knapsack_result, policy_comparison
):
    knapsack_result["estimated_roi"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        create_strategy_summary(
            knapsack_result,
            policy_comparison,
            dataset_name="Test Dataset",
        )


def test_strategy_summary_handles_zero_budget(policy_comparison):
    result = {
        "n_targeted": 0,
        "total_cost": 0.0,
        "budget_limit": 0.0,
        "total_incremental_value": 0.0,
        "estimated_roi": 0.0,
    }

    summary = create_strategy_summary(
        result,
        policy_comparison,
        dataset_name="Zero Budget Test",
    )

    assert summary["budget_utilization_pct"] == 0.0
    assert summary["budget_remaining"] == 0.0
