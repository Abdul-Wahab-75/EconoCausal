"""Budget-constrained discount allocation and optimization algorithms."""
from .knapsack import solve_knapsack_discount_allocation
from .policy_allocator import compare_allocation_policies

__all__ = [
    "solve_knapsack_discount_allocation",
    "compare_allocation_policies",
]
