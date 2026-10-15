from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from ..utils.config import Config
from ..utils.logger import get_logger
from .synthetic import generate_synthetic_data

logger = get_logger("DataLoader")


def load_dataset(
    filepath: Optional[Path] = None,
    dataset_type: str = "synthetic",
    n_synthetic_samples: int = 10000
) -> pd.DataFrame:
    """
    Loads dataset from raw path or generates synthetic benchmark data.

    Supported dataset_types:
    - 'synthetic': Generates simulated retail marketing data
    - 'rossmann': Loads and merges Rossmann store sales + store metadata
    - 'custom': Loads arbitrary CSV from filepath
    """
    if dataset_type == "synthetic":
        logger.info(f"Generating {n_synthetic_samples} synthetic customer records...")
        return generate_synthetic_data(n_samples=n_synthetic_samples)

    if dataset_type == "rossmann":
        return load_rossmann()

    if filepath is None:
        raise ValueError("Filepath must be provided for non-synthetic dataset types.")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def load_rossmann(
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads Rossmann Store Sales data, merges train.csv with store.csv,
    and maps columns to our standard causal modelling schema:

    Standard Schema Mapping:
        Promo       → Discount    (binary treatment)
        Sales       → Sales       (outcome)
        Store       → CustomerID  (unit of analysis = store)
        Customers   → footfall feature
        store.csv   → confounders (StoreType, Assortment, CompetitionDistance, etc.)

    Only open-store days with positive sales are kept (closed days are
    not meaningful for causal analysis of promotions).
    """
    if raw_dir is None:
        raw_dir = Config.RAW_DATA_DIR / "rossmann"

    train_path = raw_dir / "train.csv"
    store_path = raw_dir / "store.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Rossmann train.csv not found at {train_path}. "
            "Download from https://www.kaggle.com/c/rossmann-store-sales/data"
        )

    logger.info("Loading Rossmann train.csv...")
    train = pd.read_csv(train_path, low_memory=False)

    # Keep only open stores with positive sales
    initial = len(train)
    train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    logger.info(f"Filtered to open stores with sales > 0: {initial} -> {len(train)} rows")

    # Parse date features (these are strong confounders — promos correlate with day-of-week)
    train["Date"] = pd.to_datetime(train["Date"])
    train["Year"] = train["Date"].dt.year
    train["Month"] = train["Date"].dt.month
    train["WeekOfYear"] = train["Date"].dt.isocalendar().week.astype(int)
    train["DayOfMonth"] = train["Date"].dt.day

    # Merge store metadata (confounders: store type, assortment, competition)
    if store_path.exists():
        logger.info("Merging store.csv metadata...")
        store = pd.read_csv(store_path)

        # Fill missing competition distance with median
        store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
            store["CompetitionDistance"].median()
        )

        # Encode categorical store features as numbers
        store_type_map = {"a": 1, "b": 2, "c": 3, "d": 4}
        assortment_map = {"a": 1, "b": 2, "c": 3}
        store["StoreTypeCode"] = store["StoreType"].map(store_type_map).fillna(0).astype(int)
        store["AssortmentCode"] = store["Assortment"].map(assortment_map).fillna(0).astype(int)

        # Promo2 = continuous/recurring promo (a confounder, not the treatment)
        store["Promo2"] = store["Promo2"].fillna(0).astype(int)

        merge_cols = [
            "Store", "StoreTypeCode", "AssortmentCode",
            "CompetitionDistance", "Promo2"
        ]
        train = train.merge(store[merge_cols], on="Store", how="left")
    else:
        logger.warning("store.csv not found — proceeding without store metadata.")

    # Encode StateHoliday as numeric
    holiday_map = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}
    train["StateHoliday"] = train["StateHoliday"].map(holiday_map).fillna(0).astype(int)

    # Rename to our standard causal schema
    train = train.rename(columns={
        "Promo": "Discount",       # binary treatment
        "Store": "CustomerID",     # unit of analysis (store-day)
    })

    # Select final columns in clean order
    feature_cols = [
        "CustomerID", "Date",
        # Outcome
        "Sales",
        # Treatment
        "Discount",
        # Temporal confounders (promos are scheduled on specific days)
        "DayOfWeek", "Month", "Year", "WeekOfYear", "DayOfMonth",
        # Footfall (strong confounder — more customers → more sales)
        "Customers",
        # Store-level confounders
        "StoreTypeCode", "AssortmentCode", "CompetitionDistance", "Promo2",
        # Calendar confounders
        "StateHoliday", "SchoolHoliday",
    ]

    # Only keep columns that exist (in case store.csv was missing)
    available = [c for c in feature_cols if c in train.columns]
    train = train[available].copy()

    logger.info(
        f"Rossmann dataset ready: {len(train)} rows, "
        f"{train['Discount'].sum()} promo days, "
        f"{(train['Discount'] == 0).sum()} control days"
    )

    return train
