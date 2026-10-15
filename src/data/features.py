import pandas as pd
import numpy as np
from typing import List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs business and behavioral features for causal modeling:
    - Log transforms for skewed monetary outcomes
    - Spend-per-visit interaction features
    - Loyalty segments
    """
    data = df.copy()
    
    if "PriorSpend" in data.columns and "VisitFrequency" in data.columns:
        data["SpendPerVisit"] = data["PriorSpend"] / (data["VisitFrequency"] + 1e-5)
        
    if "Age" in data.columns:
        data["AgeGroup"] = pd.cut(
            data["Age"],
            bins=[0, 25, 40, 60, 100],
            labels=["Young", "Adult", "MiddleAged", "Senior"]
        )
        
    return data
