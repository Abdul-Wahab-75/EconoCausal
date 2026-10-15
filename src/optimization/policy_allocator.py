import numpy as np
import pandas as pd
from typing import Dict, Any
from .knapsack import solve_knapsack_discount_allocation

def compare_allocation_policies(
    df: pd.DataFrame,
    ite_estimates: np.ndarray,
    budget: float,
    discount_rate: float = 0.15
) -> pd.DataFrame:
    """
    Compares 3 different marketing allocation strategies under identical budget:
    1. Causal DML Policy (Knapsack on estimated ITE)
    2. High-Spend Heuristic (Traditional marketing: discount top historical spenders)
    3. Random Selection (Uniform random targeting)
    """
    prices = df["EstimatedItemPrice"].values if "EstimatedItemPrice" in df.columns else np.full(len(df), 50.0)
    
    # Policy 1: Causal Optimization
    causal_res = solve_knapsack_discount_allocation(
        ite_estimates=ite_estimates,
        item_prices=prices,
        budget=budget,
        discount_rate=discount_rate,
        method="milp"
    )
    
    costs = causal_res["customer_costs"]
    incremental_vals = causal_res["customer_values"]
    
    # Policy 2: High Historical Spend Heuristic
    if "PriorSpend" in df.columns:
        spend_ranks = np.argsort(-df["PriorSpend"].values)
    else:
        spend_ranks = np.argsort(-prices)
        
    heuristic_selected = np.zeros(len(df), dtype=int)
    h_cost = 0.0
    for idx in spend_ranks:
        if h_cost + costs[idx] <= budget:
            heuristic_selected[idx] = 1
            h_cost += costs[idx]
        if h_cost >= budget:
            break
            
    h_val = float(np.sum(heuristic_selected * incremental_vals))
    h_roi = (h_val / h_cost) if h_cost > 0 else 0.0
    
    # Policy 3: Random Targeting
    rng = np.random.RandomState(42)
    rand_order = rng.permutation(len(df))
    random_selected = np.zeros(len(df), dtype=int)
    r_cost = 0.0
    for idx in rand_order:
        if r_cost + costs[idx] <= budget:
            random_selected[idx] = 1
            r_cost += costs[idx]
        if r_cost >= budget:
            break
            
    r_val = float(np.sum(random_selected * incremental_vals))
    r_roi = (r_val / r_cost) if r_cost > 0 else 0.0
    
    comparison = pd.DataFrame([
        {
            "Policy": "Causal DML Optimal",
            "Targeted_Customers": int(np.sum(causal_res["selected_flags"])),
            "Budget_Spent": round(causal_res["total_cost"], 2),
            "Incremental_Gain": round(causal_res["total_incremental_value"], 2),
            "ROI": round(causal_res["estimated_roi"], 2),
            "Lift_vs_Random (%)": round(((causal_res["total_incremental_value"] - r_val) / (r_val + 1e-6)) * 100, 1)
        },
        {
            "Policy": "High-Spender Heuristic",
            "Targeted_Customers": int(np.sum(heuristic_selected)),
            "Budget_Spent": round(h_cost, 2),
            "Incremental_Gain": round(h_val, 2),
            "ROI": round(h_roi, 2),
            "Lift_vs_Random (%)": round(((h_val - r_val) / (r_val + 1e-6)) * 100, 1)
        },
        {
            "Policy": "Random Targeting",
            "Targeted_Customers": int(np.sum(random_selected)),
            "Budget_Spent": round(r_cost, 2),
            "Incremental_Gain": round(r_val, 2),
            "ROI": round(r_roi, 2),
            "Lift_vs_Random (%)": 0.0
        }
    ])
    
    return comparison
