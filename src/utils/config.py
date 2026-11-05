from pathlib import Path

class Config:
    """Project-wide paths and hyperparameter defaults."""
    # Base Directories
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    # Default Column Definitions
    TREATMENT_COL = "Discount"
    OUTCOME_COL = "Sales"
    ID_COL = "CustomerID"
    
    # Causal Modeling Defaults
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Optimization Defaults
    DEFAULT_BUDGET = 5000.0
    DEFAULT_DISCOUNT_RATE = 0.15
