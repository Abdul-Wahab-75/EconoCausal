import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from ..utils.logger import get_logger

logger = get_logger("KnapsackOptimizer")

def solve_knapsack_discount_allocation(
    ite_estimates: np.ndarray,
    item_prices: np.ndarray,
    budget: float,
    discount_rate: float = 0.15,
    method: str = "milp"
) -> Dict[str, Any]:
    """
    Solves 0-1 Knapsack problem for budget-constrained discount allocation.
    
    Formulation:
        Maximize   sum( v_i * t_i )
        Subject to sum( c_i * t_i ) <= Budget
                   t_i in {0, 1}
                   
    where:
        v_i = estimated incremental value = ITE_i * price_i
        c_i = cost of discount = price_i * discount_rate
    """
    n_customers = len(ite_estimates)
    # v_i represents the incremental revenue directly attributable to the discount (ITE)
    v = np.maximum(ite_estimates, 0.0)
    c = np.maximum(item_prices * discount_rate, 0.50)        # Cost per customer discount
    
    logger.info(f"Setting up Knapsack for {n_customers} candidates with budget ${budget:,.2f}...")

    if method == "milp":
        # Exact Mixed-Integer Linear Programming via SciPy HiGHS solver
        constraints = LinearConstraint(A=[c], lb=[0], ub=[budget])
        integrality = np.ones(n_customers)  # All 1 => binary {0, 1}
        bounds = Bounds(0, 1)
        res = milp(c=-v, constraints=constraints, integrality=integrality, bounds=bounds)
        if not res.success:
            logger.warning(f"MILP solver warning: {res.status}. Falling back to greedy.")
            selected = _greedy_knapsack(v, c, budget)
        else:
            selected = (res.x > 0.5).astype(int)
    elif method == "lp_relaxed":
        # Continuous linear programming relaxation
        res = linprog(c=-v, A_ub=[c], b_ub=[budget], bounds=[(0, 1)] * n_customers, method="highs")
        selected = (res.x > 0.5).astype(int)
    elif method == "greedy":
        selected = _greedy_knapsack(v, c, budget)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    total_targeted = int(np.sum(selected))
    total_cost = float(np.sum(selected * c))
    total_incremental_gain = float(np.sum(selected * v))
    estimated_roi = (total_incremental_gain / total_cost) if total_cost > 0 else 0.0

    logger.info(
        f"Optimization complete: Targeted {total_targeted} customers | "
        f"Cost: ${total_cost:,.2f} / ${budget:,.2f} | Incremental Value: ${total_incremental_gain:,.2f} | "
        f"ROI: {estimated_roi:.2f}x"
    )

    return {
        "selected_flags": selected,
        "n_targeted": total_targeted,
        "total_cost": total_cost,
        "budget_limit": budget,
        "total_incremental_value": total_incremental_gain,
        "estimated_roi": estimated_roi,
        "customer_values": v,
        "customer_costs": c
    }

def _greedy_knapsack(v: np.ndarray, c: np.ndarray, budget: float) -> np.ndarray:
    """Greedy approximation: sorts by value-to-cost ratio (v_i / c_i)."""
    ratios = np.divide(v, c, out=np.zeros_like(v), where=c > 0)
    sorted_indices = np.argsort(-ratios)
    
    selected = np.zeros(len(v), dtype=int)
    accumulated_cost = 0.0
    for idx in sorted_indices:
        if accumulated_cost + c[idx] <= budget:
            selected[idx] = 1
            accumulated_cost += c[idx]
        if accumulated_cost >= budget:
            break
    return selected
