import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ..utils.logger import get_logger

logger = get_logger("RobustnessChecks")

def run_placebo_test(
    estimator_class,
    X: pd.DataFrame,
    D: pd.Series,
    Y: pd.Series,
    n_iterations: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Placebo test: Randomly permutes discount indicator D.
    If causal model is sound, placebo effect should be statistically indistinguishable from zero.
    """
    rng = np.random.RandomState(random_state)
    placebo_ates = []
    
    logger.info(f"Running {n_iterations} iterations of placebo test (randomized treatment)...")
    for i in range(n_iterations):
        D_perm = pd.Series(rng.permutation(D.values), index=D.index)
        est = estimator_class(n_splits=3, nuisance_type="random_forest", backend="native", random_state=random_state + i)
        est.fit(X, D_perm, Y)
        placebo_ates.append(est.ate_)
        
    mean_placebo = float(np.mean(placebo_ates))
    std_placebo = float(np.std(placebo_ates))
    
    passed = abs(mean_placebo) < 1.0  # close to zero
    logger.info(f"Placebo Test: Mean Effect = {mean_placebo:.4f} (std={std_placebo:.4f}) | Passed: {passed}")
    return {
        "passed": passed,
        "mean_placebo_ate": mean_placebo,
        "std_placebo_ate": std_placebo,
        "iterations": placebo_ates
    }

def check_covariate_balance(
    X: pd.DataFrame,
    D: pd.Series,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculates Standardized Mean Differences (SMD) between treated and control groups.
    SMD < 0.1 indicates good balance.
    """
    treated = X[D == 1]
    control = X[D == 0]
    
    records = []
    for col in feature_names:
        if col in X.columns:
            m_t, var_t = treated[col].mean(), treated[col].var()
            m_c, var_c = control[col].mean(), control[col].var()
            pooled_std = np.sqrt((var_t + var_c) / 2.0) + 1e-6
            smd = abs(m_t - m_c) / pooled_std
            records.append({
                "Feature": col,
                "Treated_Mean": round(m_t, 3),
                "Control_Mean": round(m_c, 3),
                "SMD": round(smd, 4),
                "Balanced": smd < 0.10
            })
            
    return pd.DataFrame(records)
