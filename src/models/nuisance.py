from typing import Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

def get_nuisance_models(model_type: str = "random_forest") -> Tuple[Any, Any]:
    """
    Factory to return first-stage nuisance models:
    - model_y: predicts E[Y|X] (Sales given features)
    - model_t: predicts E[D|X] (Discount propensity given features)
    """
    if model_type == "random_forest":
        model_y = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    elif model_type == "gradient_boosting":
        model_y = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    elif model_type == "linear":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV, LogisticRegression
        model_y = make_pipeline(StandardScaler(), RidgeCV())
        model_t = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=500))
    else:
        raise ValueError(f"Unknown nuisance model_type: {model_type}")
        
    return model_y, model_t
