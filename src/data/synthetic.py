import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_data(
    n_samples: int = 10000,
    random_state: int = 42,
    noise_std: float = 15.0
) -> pd.DataFrame:
    """
    Generates synthetic customer data with known causal effects and confounding.
    
    Data Generation Process:
    - Features X: age, prior_spend, visit_frequency, loyalty_score
    - Treatment D: Bernoulli(sigmoid(w^T X)) [Confounded treatment assignment]
    - Outcome Y: baseline g(X) + theta(X) * D + epsilon
    - Returns DataFrame including true_ite for validation.
    """
    rng = np.random.RandomState(random_state)
    
    # Customer features
    customer_id = np.arange(1, n_samples + 1)
    age = rng.normal(loc=38, scale=12, size=n_samples).clip(18, 75)
    prior_spend = rng.exponential(scale=120, size=n_samples).clip(10, 1000)
    visit_frequency = rng.poisson(lam=4, size=n_samples).clip(1, 30)
    loyalty_score = rng.beta(a=2, b=5, size=n_samples) * 10.0  # 0 to 10
    
    # Confounded propensity to receive discount:
    # Marketers target less frequent or lower-spending users more often
    z = (
        -0.5
        - 0.015 * (age - 38)
        - 0.003 * (prior_spend - 120)
        - 0.15 * (visit_frequency - 4)
        + 0.10 * (5.0 - loyalty_score)
    )
    propensity = 1.0 / (1.0 + np.exp(-z))
    propensity = np.clip(propensity, 0.05, 0.95)  # Enforce overlap/positivity
    discount = rng.binomial(1, propensity, size=n_samples)
    
    # Baseline expected purchases without discount
    baseline_sales = (
        45.0
        + 1.2 * age
        + 0.45 * prior_spend
        + 8.5 * visit_frequency
        + 12.0 * loyalty_score
    )
    
    # Heterogeneous Treatment Effect (ITE)
    # Higher uplift for price-sensitive (lower prior spend) and moderately frequent users
    true_ite = (
        25.0
        + 0.05 * age
        - 0.02 * prior_spend
        + 1.8 * visit_frequency
    ).clip(5.0, 75.0)
    
    # Observed outcome
    noise = rng.normal(0, noise_std, size=n_samples)
    sales = baseline_sales + true_ite * discount + noise
    sales = np.maximum(sales, 0.0)
    
    # Customer price proxy for knapsack optimization
    price = 0.5 * baseline_sales + rng.uniform(10, 30, size=n_samples)
    
    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Age": np.round(age, 1),
        "PriorSpend": np.round(prior_spend, 2),
        "VisitFrequency": visit_frequency,
        "LoyaltyScore": np.round(loyalty_score, 2),
        "Discount": discount,
        "Sales": np.round(sales, 2),
        "EstimatedItemPrice": np.round(price, 2),
        "TrueITE": np.round(true_ite, 2),
        "TruePropensity": np.round(propensity, 4),
    })
    
    return df
