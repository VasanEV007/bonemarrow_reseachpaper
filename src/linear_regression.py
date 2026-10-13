"""
linear_regression.py - Model 1: Baseline OLS Linear Regression & Bootstrapping
Includes OLS estimation, residual analysis, error metrics, and bootstrap parameter confidence intervals.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm


class LinearRegressionPipeline:
    """
    Ordinary Least Squares (OLS) Linear Regression Pipeline with Bootstrapped Parameter Estimation.
    """

    def __init__(self, n_bootstrap: int = 100, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.model = LinearRegression()
        self.sm_model = None
        self.feature_names = []
        self.bootstrap_coefs_df = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionPipeline":
        """Fit OLS model and perform bootstrapping for confidence intervals."""
        self.feature_names = list(X_train.columns)
        X_arr = X_train.values
        y_arr = y_train.values

        # 1. Standard scikit-learn OLS
        self.model.fit(X_arr, y_arr)

        # 2. Statsmodels OLS for diagnostic summary & p-values
        import warnings
        X_sm = sm.add_constant(X_arr)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sm_model = sm.OLS(y_arr, X_sm).fit()
        except Exception:
            self.sm_model = None

        # 3. Bootstrapping parameter estimates (resampling 50-100 times with replacement)
        np.random.seed(self.random_state)
        n_samples = len(X_arr)
        boot_intercepts = []
        boot_slopes = []

        for b in range(self.n_bootstrap):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b = X_arr[boot_idx]
            y_b = y_arr[boot_idx]
            reg_b = LinearRegression()
            reg_b.fit(X_b, y_b)
            boot_intercepts.append(reg_b.intercept_)
            boot_slopes.append(reg_b.coef_)

        boot_slopes = np.array(boot_slopes)  # shape (B, p)
        boot_intercepts = np.array(boot_intercepts)  # shape (B,)

        # Build summary DataFrame for bootstrapped parameters
        rows = []
        # Intercept
        rows.append({
            "Feature": "Intercept",
            "Mean_Coefficient": float(np.mean(boot_intercepts)),
            "Std_Error": float(np.std(boot_intercepts)),
            "CI_Lower_95": float(np.percentile(boot_intercepts, 2.5)),
            "CI_Upper_95": float(np.percentile(boot_intercepts, 97.5)),
            "P_Value": float(self.sm_model.pvalues[0]) if self.sm_model is not None else np.nan,
        })

        for i, col_name in enumerate(self.feature_names):
            coefs = boot_slopes[:, i]
            p_val = self.sm_model.pvalues[i + 1] if self.sm_model is not None and len(self.sm_model.pvalues) > i + 1 else np.nan
            rows.append({
                "Feature": col_name,
                "Mean_Coefficient": float(np.mean(coefs)),
                "Std_Error": float(np.std(coefs)),
                "CI_Lower_95": float(np.percentile(coefs, 2.5)),
                "CI_Upper_95": float(np.percentile(coefs, 97.5)),
                "P_Value": float(p_val),
            })

        self.bootstrap_coefs_df = pd.DataFrame(rows)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both continuous values and binary classification thresholded at 0.5.
        Returns: (y_continuous, y_binary)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict!")
        X_arr = X_test.values
        y_cont = self.model.predict(X_arr)
        # Probabilistic classification threshold
        y_bin = (y_cont >= 0.5).astype(int)
        return y_cont, y_bin

    def evaluate_regression(self, y_true: np.ndarray, y_pred_cont: np.ndarray) -> Dict[str, float]:
        """Compute standard regression error metrics: MAE, MSE, RMSE, R^2."""
        mae = mean_absolute_error(y_true, y_pred_cont)
        mse = mean_squared_error(y_true, y_pred_cont)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_cont)
        return {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2_Score": float(r2),
        }

    def compute_residuals_table(
        self, sample_ids: pd.Series, y_true: np.ndarray, y_pred_cont: np.ndarray
    ) -> pd.DataFrame:
        """Construct detailed table of regression residuals and errors."""
        res = y_true - y_pred_cont
        return pd.DataFrame({
            "Sample_ID": sample_ids.values,
            "Actual_Survival": y_true,
            "Predicted_Continuous": np.round(y_pred_cont, 4),
            "Residual": np.round(res, 4),
            "Absolute_Error": np.round(np.abs(res), 4),
            "Squared_Error": np.round(res ** 2, 4),
        })
