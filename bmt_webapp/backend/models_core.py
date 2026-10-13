"""
models_core.py - Core Model Classes and Preprocessor Definitions
Ensures stable module paths for joblib/pickle serialization.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ClinicalBMTPreprocessor:
    """
    Dedicated Preprocessor for the 35 Clinical Input Features.
    Strictly fitted on training data; transforms test and new patient inputs without data leakage.
    """

    def __init__(self, iqr_factor: float = 3.0):
        self.iqr_factor = iqr_factor
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = ["Disease"]
        self.num_medians = {}
        self.cat_modes = {}
        self.outlier_bounds = {}
        self.scaler = StandardScaler()
        self.disease_categories = []
        self.transformed_feature_names = []
        self.is_fitted = False

    @staticmethod
    def standardize_disease(val):
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

    def fit(self, df_train: pd.DataFrame):
        df = df_train.copy().replace("?", np.nan)
        exclude = ["id", "survival_time", "survival_status"]
        self.feature_cols = [c for c in df.columns if c not in exclude]

        df["Disease"] = df["Disease"].apply(self.standardize_disease)
        self.numeric_cols = [c for c in self.feature_cols if c != "Disease"]

        for col in self.numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            med = float(s.median()) if pd.notnull(s.median()) else 0.0
            self.num_medians[col] = med

            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            if pd.notnull(iqr) and iqr > 0:
                lower = q25 - self.iqr_factor * iqr
                upper = q75 + self.iqr_factor * iqr
            else:
                lower = s.min() if pd.notnull(s.min()) else 0.0
                upper = s.max() if pd.notnull(s.max()) else 1.0
            self.outlier_bounds[col] = (float(lower), float(upper))

        mode_val = df["Disease"].dropna().mode()
        self.cat_modes["Disease"] = mode_val.iloc[0] if len(mode_val) > 0 else "all"

        self.disease_categories = sorted(list(df["Disease"].fillna(self.cat_modes["Disease"]).unique()))

        temp_num = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians[col])
            lower, upper = self.outlier_bounds[col]
            temp_num[col] = vals.clip(lower=lower, upper=upper)

        self.scaler.fit(temp_num)

        self.transformed_feature_names = list(self.numeric_cols) + [
            f"Disease_{c}" for c in self.disease_categories[1:]
        ]
        self.is_fitted = True
        return self

    def transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first!")

        df = df_input.copy().replace("?", np.nan)
        if "Disease" in df.columns:
            df["Disease"] = df["Disease"].apply(self.standardize_disease)

        imputed_num = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians[col])
            else:
                vals = pd.Series(self.num_medians[col], index=df.index)
            lower, upper = self.outlier_bounds[col]
            imputed_num[col] = vals.clip(lower=lower, upper=upper)

        scaled_arr = self.scaler.transform(imputed_num)
        scaled_df = pd.DataFrame(scaled_arr, columns=self.numeric_cols, index=df.index)

        disease_series = df["Disease"].fillna(self.cat_modes["Disease"]) if "Disease" in df.columns else pd.Series(self.cat_modes["Disease"], index=df.index)
        encoded_cat = pd.DataFrame(index=df.index)
        for cat in self.disease_categories[1:]:
            encoded_cat[f"Disease_{cat}"] = (disease_series == cat).astype(float)

        return pd.concat([scaled_df, encoded_cat], axis=1)


class PNNProbabilityExpander:
    """Stage 1: PNN Probability Expansion across Euclidean, Chebyshev, and Cosine metrics."""

    def __init__(self, sigma: float = 3.2183):
        self.sigma = max(sigma, 1e-4)
        self.X_train = None
        self.y_train = None
        self.idx_class0 = None
        self.idx_class1 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.idx_class0 = np.where(self.y_train == 0)[0]
        self.idx_class1 = np.where(self.y_train == 1)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        eps = 1e-9
        var = 2.0 * (self.sigma ** 2)

        d_e = cdist(X, self.X_train, metric="euclidean")
        k_e = np.exp(- (d_e ** 2) / var)

        d_c = cdist(X, self.X_train, metric="chebyshev")
        k_c = np.exp(- (d_c ** 2) / var)

        d_cos = np.nan_to_num(cdist(X, self.X_train, metric="cosine"), nan=1.0)
        k_cos = np.exp(- (d_cos ** 2) / var)

        probs = np.zeros((n, 3), dtype=np.float64)
        for m_idx, K in enumerate([k_e, k_c, k_cos]):
            sum0 = np.mean(K[:, self.idx_class0], axis=1) if len(self.idx_class0) > 0 else np.zeros(n)
            sum1 = np.mean(K[:, self.idx_class1], axis=1) if len(self.idx_class1) > 0 else np.zeros(n)
            probs[:, m_idx] = sum1 / (sum0 + sum1 + eps)
        return probs

    def expand_features(self, X: np.ndarray) -> np.ndarray:
        probs = self.transform(X)
        return np.hstack([X, probs])


class CascadeClassifierAppModel:
    """
    PNN + Random Forest Cascade-Based Input-Doubling Classifier (Izonin et al., 2025).
    Features probabilistic plurality voting across support vectors to deliver continuous,
    sensitive clinical mortality risk assessments.
    """

    def __init__(
        self,
        sigma: float = 3.2183,
        n_estimators: int = 100,
        max_depth: int = 9,
        max_pairs_train: int = 6000,
        n_reference_voting: int = 40,
        random_state: int = 42,
    ):
        self.sigma = sigma
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_pairs_train = max_pairs_train
        self.n_reference_voting = n_reference_voting
        self.random_state = random_state

        self.pnn = PNNProbabilityExpander(sigma=self.sigma)
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.X_train_expanded = None
        self.y_train = None
        self.ref_indices = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.int64)
        n = len(X_arr)

        self.pnn.fit(X_arr, y_arr)
        self.X_train_expanded = self.pnn.expand_features(X_arr)
        self.y_train = y_arr

        np.random.seed(self.random_state)
        n_ref = min(self.n_reference_voting, n)
        self.ref_indices = np.random.choice(n, size=n_ref, replace=False)

        pair_i, pair_j = [], []
        for i in range(n):
            js = np.random.choice([x for x in range(n) if x != i], size=min(n - 1, 55), replace=False)
            for j in js:
                pair_i.append(i)
                pair_j.append(j)

        pair_i = np.array(pair_i)
        pair_j = np.array(pair_j)

        if len(pair_i) > self.max_pairs_train:
            sub = np.random.choice(len(pair_i), size=self.max_pairs_train, replace=False)
            pair_i, pair_j = pair_i[sub], pair_j[sub]

        X_doubled = np.hstack([self.X_train_expanded[pair_i], self.X_train_expanded[pair_j]])
        z_targets = self.y_train[pair_i] - self.y_train[pair_j]

        self.rf.fit(X_doubled, z_targets)
        self.is_fitted = True
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first!")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        n_test = len(X_test_arr)
        X_test_exp = self.pnn.expand_features(X_test_arr)

        ref_exp = self.X_train_expanded[self.ref_indices]
        ref_y = self.y_train[self.ref_indices]
        n_ref = len(ref_y)

        test_rep = np.repeat(X_test_exp, n_ref, axis=0)
        ref_tile = np.tile(ref_exp, (n_test, 1))
        X_doubled_query = np.hstack([test_rep, ref_tile])

        # Tree ensemble probabilistic voting for difference classes {-1, 0, 1}
        classes = list(self.rf.classes_)
        idx_0 = classes.index(0) if 0 in classes else None
        idx_pos1 = classes.index(1) if 1 in classes else None

        z_probs = self.rf.predict_proba(X_doubled_query)
        p_pos1 = z_probs[:, idx_pos1].reshape(n_test, n_ref) if idx_pos1 is not None else np.zeros((n_test, n_ref))
        p_zero = z_probs[:, idx_0].reshape(n_test, n_ref) if idx_0 is not None else np.zeros((n_test, n_ref))

        # Plurality voting across reference training vectors:
        # If reference label y_ref == 0, query is class 1 when difference z = +1
        # If reference label y_ref == 1, query is class 1 when difference z = 0
        ref_y_matrix = np.tile(ref_y, (n_test, 1))
        vote_probs = np.where(ref_y_matrix == 0, p_pos1, p_zero)

        p1 = np.mean(vote_probs, axis=1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        prob1 = self.predict_proba(X_test)[:, 1]
        return (prob1 >= 0.5).astype(int)
