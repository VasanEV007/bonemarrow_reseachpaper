"""
data_cleaning.py - Data Cleaning & Preprocessing Workflow
Handles missing values, deduplication, string standardization, outlier clipping, encoding, and scaling.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BoneMarrowPreprocessor:
    """
    Production-ready Preprocessor for Bone Marrow Transplant Dataset.
    Fits imputers, outlier caps, encoders, and scalers strictly on the training set
    and applies them consistently across train and test sets without data leakage.
    """

    def __init__(self, iqr_factor: float = 3.0):
        self.iqr_factor = iqr_factor
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.num_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.scaler = StandardScaler()
        self.disease_categories: List[str] = []
        self.is_fitted = False

    @staticmethod
    def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize messy string formatting and casing in categorical columns."""
        df_clean = df.copy()
        if "Disease" in df_clean.columns:
            def clean_disease(val):
                if pd.isnull(val):
                    return "unknown"
                s = str(val).strip().lower().replace("-", "")
                if "non" in s:
                    return "nonmalignant"
                if "aml" in s:
                    return "aml"
                if "all" in s:
                    return "all"
                if "chron" in s:
                    return "chronic"
                if "lymph" in s:
                    return "lymphoma"
                return s
            df_clean["Disease"] = df_clean["Disease"].apply(clean_disease)
        return df_clean

    def fit(self, df_train: pd.DataFrame) -> "BoneMarrowPreprocessor":
        """Fit cleaning parameters (medians, modes, outlier bounds, scaler) on training data."""
        df = df_train.copy()
        df = df.replace("?", np.nan)
        df = self.standardize_strings(df)

        # Drop duplicates before fitting parameters
        df = df.drop_duplicates()

        feature_cols = [c for c in df.columns if c not in ["id", "survival_status"]]

        # Separate categorical vs numerical
        self.categorical_cols = ["Disease"]
        self.numeric_cols = [c for c in feature_cols if c not in self.categorical_cols]

        # 1. Compute numerical medians and IQR outlier bounds
        for col in self.numeric_cols:
            col_series = pd.to_numeric(df[col], errors="coerce")
            median_val = col_series.median()
            if pd.isnull(median_val):
                median_val = 0.0
            self.num_medians[col] = float(median_val)

            # Compute IQR bounds
            q25 = col_series.quantile(0.25)
            q75 = col_series.quantile(0.75)
            iqr = q75 - q25
            if pd.notnull(iqr) and iqr > 0:
                lower = q25 - self.iqr_factor * iqr
                upper = q75 + self.iqr_factor * iqr
            else:
                lower = col_series.min() if pd.notnull(col_series.min()) else 0.0
                upper = col_series.max() if pd.notnull(col_series.max()) else 1.0
            self.outlier_bounds[col] = (float(lower), float(upper))

        # 2. Compute categorical modes
        for col in self.categorical_cols:
            mode_series = df[col].dropna().mode()
            self.cat_modes[col] = mode_series.iloc[0] if len(mode_series) > 0 else "unknown"

        # Unique disease categories for one-hot encoding
        disease_clean = df["Disease"].fillna(self.cat_modes["Disease"])
        self.disease_categories = sorted(list(disease_clean.unique()))

        # Pre-fit standard scaler on imputed and outlier-clipped numerical columns
        temp_num = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians[col])
            lower, upper = self.outlier_bounds[col]
            vals = vals.clip(lower=lower, upper=upper)
            temp_num[col] = vals

        self.scaler.fit(temp_num)
        self.is_fitted = True
        return self

    def transform(self, df_input: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Applies cleaning, deduplication (if train), imputation, outlier clipping,
        one-hot encoding, and scaling to the input dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform!")

        df = df_input.copy()
        df = df.replace("?", np.nan)
        df = self.standardize_strings(df)

        if is_train:
            # Deduplication
            df = df.drop_duplicates()

        # Handle numerical missing values and outlier capping
        imputed_num = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians[col])
            lower, upper = self.outlier_bounds[col]
            vals = vals.clip(lower=lower, upper=upper)
            imputed_num[col] = vals

        # Scale continuous features
        scaled_num_arr = self.scaler.transform(imputed_num)
        scaled_num_df = pd.DataFrame(scaled_num_arr, columns=self.numeric_cols, index=df.index)

        # One-hot encode Disease (drop first category to avoid dummy variable trap / singular matrix in OLS)
        disease_col = df["Disease"].fillna(self.cat_modes["Disease"])
        encoded_cat = pd.DataFrame(index=df.index)
        # Drop the first category as baseline reference
        for cat in self.disease_categories[1:]:
            encoded_cat[f"Disease_{cat}"] = (disease_col == cat).astype(float)

        # Assemble final cleaned feature matrix
        out_df = pd.concat([scaled_num_df, encoded_cat], axis=1)

        # Re-attach id and target if present
        if "id" in df.columns:
            out_df.insert(0, "id", df["id"].values)
        if "survival_status" in df.columns:
            out_df["survival_status"] = pd.to_numeric(df["survival_status"], errors="coerce").fillna(0).astype(int).values

        return out_df
