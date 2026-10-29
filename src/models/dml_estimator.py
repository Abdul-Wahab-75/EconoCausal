
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from ..utils.logger import get_logger
from .nuisance import get_nuisance_models


logger = get_logger("DoubleMLEstimator")


class DoubleMLEstimator:
    """
    Partially Linear Double Machine Learning (DML / PLR) Estimator.

    Model:
        Y = g(X) + theta(X) * D + epsilon
        D = m(X) + eta

    Uses K-Fold Cross-Fitting to compute orthogonalized residuals:
        Y_res = Y - g_hat(X)
        D_res = D - m_hat(X)

    Estimates the average treatment effect and approximate
    heterogeneous treatment effects.
    """

    def __init__(
        self,
        n_splits: int = 5,
        nuisance_type: str = "random_forest",
        backend: str = "econml",
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.nuisance_type = nuisance_type
        self.backend = backend
        self.random_state = random_state

        self.ate_: Optional[float] = None
        self.standard_error_: Optional[float] = None
        self.ite_estimates_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        self._econml_model: Optional[Any] = None
        self._het_model: Optional[Any] = None
        self._ite_projection_mean: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        D: pd.Series,
        Y: pd.Series
    ) -> "DoubleMLEstimator":
        """Fit nuisance models and estimate causal effects."""

        logger.info(
            f"Fitting DML Estimator using backend "
            f"'{self.backend}' and '{self.nuisance_type}' nuisances..."
        )

        # Reset prediction models before fitting again.
        self._econml_model = None
        self._het_model = None
        self._ite_projection_mean = None
        self.is_fitted = False

        # Try EconML if selected and installed.
        if self.backend == "econml":
            try:
                from econml.dml import LinearDML

                model_y, model_t = get_nuisance_models(
                    self.nuisance_type
                )

                self._econml_model = LinearDML(
                    model_y=model_y,
                    model_t=model_t,
                    cv=self.n_splits,
                    discrete_treatment=True,
                    random_state=self.random_state
                )

                self._econml_model.fit(
                    Y=Y.values,
                    T=D.values,
                    X=X.values
                )

                self.ite_estimates_ = (
                    self._econml_model
                    .const_marginal_effect(X.values)
                    .ravel()
                )

                self.ate_ = float(
                    np.mean(self.ite_estimates_)
                )

                self.standard_error_ = float(
                    np.std(self.ite_estimates_)
                    / np.sqrt(len(self.ite_estimates_))
                )

                self.is_fitted = True

                logger.info(
                    f"EconML fit complete: "
                    f"Average Treatment Effect (ATE) = {self.ate_:.3f}"
                )

                return self

            except Exception as exc:
                self._econml_model = None
                logger.warning(
                    f"EconML fit encountered issue ({exc}). "
                    "Falling back to native cross-fitted DML."
                )

        # Native Cross-Fitted DML.
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        y_res = np.zeros(len(Y))
        d_res = np.zeros(len(D))

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        D_arr = D.values if isinstance(D, pd.Series) else np.asarray(D)
        Y_arr = Y.values if isinstance(Y, pd.Series) else np.asarray(Y)

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            Y_tr, Y_val = Y_arr[train_idx], Y_arr[val_idx]
            D_tr, D_val = D_arr[train_idx], D_arr[val_idx]

            m_y, m_t = get_nuisance_models(self.nuisance_type)

            m_y.fit(X_tr, Y_tr)
            m_t.fit(X_tr, D_tr)

            y_res[val_idx] = Y_val - m_y.predict(X_val)
            d_res[val_idx] = D_val - m_t.predict(X_val)

        # Second stage: regress Y residuals on D residuals.
        sec_stage = LinearRegression(fit_intercept=False)
        sec_stage.fit(d_res.reshape(-1, 1), y_res)

        self.ate_ = float(sec_stage.coef_[0])

        residuals = y_res - self.ate_ * d_res

        variance = (
            np.mean(residuals**2 * d_res**2)
            / (np.mean(d_res**2)**2 * len(Y_arr))
        )

        self.standard_error_ = float(np.sqrt(variance))

        # Approximate heterogeneous treatment effects.
        interacted_X = d_res.reshape(-1, 1) * X_arr

        het_model = LinearRegression(fit_intercept=True)
        het_model.fit(interacted_X, y_res)

        # Preserve the training projection mean so predictions for
        # new records use the same centering as training estimates.
        projection = X_arr @ het_model.coef_

        self._het_model = het_model
        self._ite_projection_mean = float(np.mean(projection))

        self.ite_estimates_ = (
            self.ate_
            + projection
            - self._ite_projection_mean
        )

        self.is_fitted = True

        logger.info(
            f"Native DML fit complete: ATE = {self.ate_:.3f} "
            f"+/- {1.96 * self.standard_error_:.3f}"
        )

        return self

    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """Predict individual treatment effects for supplied records."""

        if not self.is_fitted:
            raise RuntimeError(
                "Estimator must be fitted before predicting ITE."
            )

        if self._econml_model is not None:
            return np.asarray(
                self._econml_model
                .const_marginal_effect(X.values)
            ).reshape(-1)

        if self._het_model is None or self._ite_projection_mean is None:
            raise RuntimeError(
                "Native ITE prediction model is not available."
            )

        X_arr = (
            X.to_numpy()
            if isinstance(X, pd.DataFrame)
            else np.asarray(X)
        )

        if X_arr.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional feature matrix."
            )

        if X_arr.shape[1] != self._het_model.coef_.shape[0]:
            raise ValueError(
                f"Expected {self._het_model.coef_.shape[0]} features, "
                f"received {X_arr.shape[1]}."
            )

        projection = X_arr @ self._het_model.coef_

        return np.asarray(
            self.ate_
            + projection
            - self._ite_projection_mean
        ).reshape(-1)