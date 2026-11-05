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
    Then estimates treatment effect theta via Neyman-orthogonal score.
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

    def fit(self, X: pd.DataFrame, D: pd.Series, Y: pd.Series) -> "DoubleMLEstimator":
        """Fits first-stage nuisance models with cross-fitting and estimates causal effects."""
        logger.info(f"Fitting DML Estimator using backend '{self.backend}' and '{self.nuisance_type}' nuisances...")
        
        # Try EconML if selected and installed
        if self.backend == "econml":
            try:
                from econml.dml import LinearDML
                model_y, model_t = get_nuisance_models(self.nuisance_type)
                self._econml_model = LinearDML(
                    model_y=model_y,
                    model_t=model_t,
                    cv=self.n_splits,
                    discrete_treatment=True,
                    random_state=self.random_state
                )
                self._econml_model.fit(Y=Y.values, T=D.values, X=X.values)
                self.ite_estimates_ = self._econml_model.const_marginal_effect(X.values).ravel()
                self.ate_ = float(np.mean(self.ite_estimates_))
                self.standard_error_ = float(np.std(self.ite_estimates_) / np.sqrt(len(self.ite_estimates_)))
                self.is_fitted = True
                logger.info(f"EconML fit complete: Average Treatment Effect (ATE) = {self.ate_:.3f}")
                return self
            except Exception as e:
                logger.warning(f"EconML fit encountered issue ({e}). Falling back to native cross-fitted DML.")

        # Native Cross-Fitted DML (Chernozhukov et al. 2018 Robinson transformation)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        y_res = np.zeros(len(Y))
        d_res = np.zeros(len(D))
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        D_arr = D.values if isinstance(D, pd.Series) else D
        Y_arr = Y.values if isinstance(Y, pd.Series) else Y

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            Y_tr, Y_val = Y_arr[train_idx], Y_arr[val_idx]
            D_tr, D_val = D_arr[train_idx], D_arr[val_idx]

            m_y, m_t = get_nuisance_models(self.nuisance_type)
            m_y.fit(X_tr, Y_tr)
            m_t.fit(X_tr, D_tr)

            y_res[val_idx] = Y_val - m_y.predict(X_val)
            d_res[val_idx] = D_val - m_t.predict(X_val)

        # 2nd stage: Regress Y_res on D_res
        # theta = Cov(Y_res, D_res) / Var(D_res)
        sec_stage = LinearRegression(fit_intercept=False)
        sec_stage.fit(d_res.reshape(-1, 1), y_res)
        
        self.ate_ = float(sec_stage.coef_[0])
        residuals = y_res - self.ate_ * d_res
        variance = np.mean(residuals**2 * d_res**2) / (np.mean(d_res**2)**2 * len(Y_arr))
        self.standard_error_ = float(np.sqrt(variance))
        
        # Heterogeneous effect approximation based on interaction
        # For PLR with heterogeneous effects, fit D_res * X -> Y_res
        het_model = LinearRegression(fit_intercept=True)
        interacted_X = d_res.reshape(-1, 1) * X_arr
        het_model.fit(interacted_X, y_res)
        self.ite_estimates_ = self.ate_ + (X_arr @ het_model.coef_ - np.mean(X_arr @ het_model.coef_))
        
        self.is_fitted = True
        logger.info(f"Native DML fit complete: ATE = {self.ate_:.3f} +/- {1.96*self.standard_error_:.3f}")
        return self

    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """Returns estimated Individual Treatment Effects (ITE) for each customer."""
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before predicting ITE.")
        if self._econml_model is not None:
            return self._econml_model.const_marginal_effect(X.values).ravel()
        return self.ite_estimates_
