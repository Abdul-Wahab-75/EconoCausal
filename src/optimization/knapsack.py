
import numpy as np
from typing import Dict, Any
from scipy.optimize import milp, LinearConstraint, Bounds
from ..utils.logger import get_logger

logger = get_logger("KnapsackOptimizer")


def solve_knapsack_discount_allocation(
    ite_estimates,
    item_prices,
    budget: float,
    discount_rate: float = 0.15,
    method: str = "milp",
    ite_is_revenue: bool = False,
) -> Dict[str, Any]:
    """
    Maximize estimated incremental value under a fixed discount budget.

    Decision variable:
        x_i = 1 if customer i is selected, otherwise 0.

    Value:
        Revenue ITE is converted to 10% gross profit when
        ite_is_revenue=True.
        Otherwise ITE is multiplied by item price and 10%.

    Cost:
        item price * discount rate.
    """

    # Convert API lists and other inputs into NumPy arrays.
    ite_estimates = np.asarray(ite_estimates, dtype=float).reshape(-1)
    item_prices = np.asarray(item_prices, dtype=float).reshape(-1)
    budget = float(budget)
    discount_rate = float(discount_rate)

    # Validate inputs.
    if len(ite_estimates) == 0:
        raise ValueError("At least one ITE estimate is required.")

    if len(ite_estimates) != len(item_prices):
        raise ValueError(
            "ITE estimates and item prices must have equal lengths."
        )

    if not np.all(np.isfinite(ite_estimates)):
        raise ValueError("ITE estimates must contain only finite numbers.")

    if not np.all(np.isfinite(item_prices)):
        raise ValueError("Item prices must contain only finite numbers.")

    if np.any(item_prices <= 0):
        raise ValueError("All item prices must be greater than zero.")

    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be greater than zero.")

    if not 0 < discount_rate <= 1:
        raise ValueError("Discount rate must be greater than 0 and at most 1.")

    n_customers = len(ite_estimates)

    # Calculate estimated value and discount cost.
    if ite_is_revenue:
        values = np.maximum(ite_estimates * 0.10, 0.0)
    else:
        values = np.maximum(
            ite_estimates * item_prices * 0.10,
            0.0
        )

    costs = np.maximum(item_prices * discount_rate, 0.50)

    logger.info(
        "Setting up knapsack for %s candidates with budget %.2f",
        n_customers,
        budget,
    )

    if method == "milp":
        # Binary decision variables: each customer is selected (1) or not (0).
        constraint = LinearConstraint(
            A=costs.reshape(1, -1),
            lb=-np.inf,
            ub=budget,
        )

        result = milp(
            c=-values,
            integrality=np.ones(n_customers, dtype=int),
            bounds=Bounds(
                lb=np.zeros(n_customers),
                ub=np.ones(n_customers),
            ),
            constraints=constraint,
        )

        if not result.success or result.x is None:
            logger.warning(
                "MILP solver failed (%s). Using greedy fallback.",
                result.message,
            )
            selected = _greedy_knapsack(values, costs, budget)
        else:
            selected = (result.x > 0.5).astype(int)

    elif method == "greedy":
        selected = _greedy_knapsack(values, costs, budget)

    else:
        raise ValueError(
            "Unknown optimization method. Use 'milp' or 'greedy'."
        )

    total_targeted = int(np.sum(selected))
    total_cost = float(np.sum(selected * costs))
    total_incremental_gain = float(np.sum(selected * values))

    estimated_roi = (
        total_incremental_gain / total_cost
        if total_cost > 0
        else 0.0
    )

    logger.info(
        "Optimization complete: targeted=%s, cost=%.2f, "
        "incremental_value=%.2f, ROI=%.2f",
        total_targeted,
        total_cost,
        total_incremental_gain,
        estimated_roi,
    )

    return {
        "selected_flags": selected,
        "n_targeted": total_targeted,
        "total_cost": total_cost,
        "budget_limit": budget,
        "total_incremental_value": total_incremental_gain,
        "estimated_roi": estimated_roi,
        "customer_values": values,
        "customer_costs": costs,
    }


def _greedy_knapsack(
    values: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Greedy fallback based on estimated value divided by cost."""

    ratios = np.divide(
        values,
        costs,
        out=np.zeros_like(values),
        where=costs > 0,
    )

    sorted_indices = np.argsort(-ratios)
    selected = np.zeros(len(values), dtype=int)
    accumulated_cost = 0.0

    for index in sorted_indices:
        if accumulated_cost + costs[index] <= budget:
            selected[index] = 1
            accumulated_cost += costs[index]

        if accumulated_cost >= budget:
            break

    return selected