import pytest
import numpy as np
import pandas as pd
from src.optimization.knapsack import solve_knapsack_discount_allocation
from src.optimization.policy_allocator import compare_allocation_policies
from src.data.synthetic import generate_synthetic_data

def test_knapsack_solver():
    ite = np.array([10.0, 50.0, 5.0, 30.0, 2.0])
    prices = np.array([20.0, 50.0, 15.0, 40.0, 10.0])
    budget = 10.0
    
    res = solve_knapsack_discount_allocation(
        ite_estimates=ite,
        item_prices=prices,
        budget=budget,
        discount_rate=0.10,
        method="milp"
    )
    
    assert res["total_cost"] <= budget + 1e-5
    assert res["n_targeted"] >= 1
    assert len(res["selected_flags"]) == 5

def test_policy_comparison():
    df = generate_synthetic_data(n_samples=200, random_state=42)
    ite = np.random.uniform(5, 30, size=200)
    comparison = compare_allocation_policies(df, ite, budget=500.0)
    assert len(comparison) == 3
    assert "Policy" in comparison.columns
    assert "Incremental_Gain" in comparison.columns
def test_knapsack_respects_budget_and_ignores_negative_values():
    ite = np.array([10.0, -50.0, 5.0])
    prices = np.array([20.0, 50.0, 15.0])
    budget = 2.0

    result = solve_knapsack_discount_allocation(
        ite_estimates=ite,
        item_prices=prices,
        budget=budget,
        discount_rate=0.10,
        method="milp",
    )

    assert result["total_cost"] <= budget + 1e-5
    assert result["selected_flags"][1] == 0
    assert result["n_targeted"] == int(
        np.sum(result["selected_flags"])
    )