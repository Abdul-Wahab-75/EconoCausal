from pathlib import Path
from typing import Optional
import pandas as pd
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
    - 'rossmann': Loads Rossmann store sales CSV
    - 'custom': Loads arbitrary CSV from filepath
    """
    if dataset_type == "synthetic":
        logger.info(f"Generating {n_synthetic_samples} synthetic customer records...")
        return generate_synthetic_data(n_samples=n_synthetic_samples)
    
    if filepath is None:
        raise ValueError("Filepath must be provided for non-synthetic dataset types.")
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
        
    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df
