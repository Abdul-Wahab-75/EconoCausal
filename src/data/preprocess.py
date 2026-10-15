import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ..utils.logger import get_logger

logger = get_logger("Preprocess")

def clean_data(df: pd.DataFrame, drop_missing_cols: bool = True) -> pd.DataFrame:
    """Cleans dataframe by handling missing values and dropping invalid rows."""
    cleaned = df.copy()
    initial_rows = len(cleaned)
    cleaned = cleaned.dropna()
    logger.info(f"Cleaned data: dropped {initial_rows - len(cleaned)} rows with missing values.")
    return cleaned

def preprocess_features(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, StandardScaler, Optional[OneHotEncoder]]:
    """Encodes categorical columns and standardizes numerical columns."""
    df_proc = df.copy()
    
    scaler = StandardScaler()
    if numerical_cols:
        df_proc[numerical_cols] = scaler.fit_transform(df_proc[numerical_cols])
        
    encoder = None
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_cats = encoder.fit_transform(df_proc[categorical_cols])
        encoded_names = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_names, index=df_proc.index)
        df_proc = df_proc.drop(columns=categorical_cols).join(encoded_df)
        
    return df_proc, scaler, encoder
