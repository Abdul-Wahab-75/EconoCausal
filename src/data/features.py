import pandas as pd
import numpy as np
from typing import List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs business and behavioral features for causal modeling.

    Works for both synthetic data and Rossmann real data.
    Detects available columns and applies the appropriate transformations.
    """
    data = df.copy()

    # ── Synthetic data features ──
    if "PriorSpend" in data.columns and "VisitFrequency" in data.columns:
        data["SpendPerVisit"] = data["PriorSpend"] / (data["VisitFrequency"] + 1e-5)

    if "Age" in data.columns:
        data["AgeGroup"] = pd.cut(
            data["Age"],
            bins=[0, 25, 40, 60, 100],
            labels=["Young", "Adult", "MiddleAged", "Senior"]
        )

    # ── Rossmann-specific features ──

    # Weekend flag (DayOfWeek: 1=Mon ... 7=Sun in Rossmann)
    if "DayOfWeek" in data.columns:
        data["IsWeekend"] = (data["DayOfWeek"] >= 6).astype(int)

    # Log-transform of Sales (reduces skewness, useful for ML models)
    if "Sales" in data.columns:
        data["LogSales"] = np.log1p(data["Sales"])

    # Log of customers (footfall is heavily right-skewed)
    if "Customers" in data.columns:
        data["LogCustomers"] = np.log1p(data["Customers"])

    # Customers per unit competition distance (store attractiveness proxy)
    if "Customers" in data.columns and "CompetitionDistance" in data.columns:
        data["CustomersPerKmCompetition"] = data["Customers"] / (
            data["CompetitionDistance"] / 1000.0 + 0.1
        )

    # Month-based seasonality indicator (Q4 = holiday season)
    if "Month" in data.columns:
        data["IsQ4"] = (data["Month"] >= 10).astype(int)

    # Any holiday (state or school)
    if "StateHoliday" in data.columns and "SchoolHoliday" in data.columns:
        data["AnyHoliday"] = ((data["StateHoliday"] > 0) | (data["SchoolHoliday"] > 0)).astype(int)

    return data


def get_rossmann_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Returns the list of feature column names to use as confounders (X)
    for the Rossmann dataset. Excludes outcome, treatment, ID, and date columns.
    """
    exclude = {"CustomerID", "Date", "Sales", "Discount", "LogSales",
               "TrueITE", "TruePropensity", "Estimated_ITE", "EstimatedItemPrice"}
    candidates = [
        # Core confounders
        "DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
        "Customers", "StoreTypeCode", "AssortmentCode",
        "CompetitionDistance", "Promo2",
        "StateHoliday", "SchoolHoliday",
        # Engineered features
        "IsWeekend", "LogCustomers", "CustomersPerKmCompetition",
        "IsQ4", "AnyHoliday",
    ]
    return [c for c in candidates if c in df.columns and c not in exclude]
