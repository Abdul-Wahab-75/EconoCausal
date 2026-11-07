from typing import Tuple, Any

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from lightgbm import LGBMRegressor, LGBMClassifier


def get_nuisance_models(model_type: str = "random_forest") -> Tuple[Any, Any]:
    """
    Return first-stage nuisance models.

    model_y: predicts E[Y|X] (Sales given features)
    model_t: predicts P(D=1|X) for binary treatment Discount.
    """

    if model_type == "random_forest":
        model_y = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

        model_t = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

    elif model_type == "gradient_boosting":
        model_y = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )

        model_t = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )

    elif model_type == "lightgbm":
        model_y = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )

        model_t = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )

    elif model_type == "linear":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV, LogisticRegression

        model_y = make_pipeline(
            StandardScaler(),
            RidgeCV()
        )

        model_t = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=42,
                max_iter=500
            )
        )

    else:
        raise ValueError(
            f"Unknown nuisance model_type: {model_type}"
        )

    return model_y, model_t
