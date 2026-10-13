"""
cascade_classifier.py - Model 2: Cascade-Based Input-Doubling Classifier (Izonin et al., 2025)
Implements Algorithm 1:
1. PNN Probability Expansion (Euclidean, Chebyshev, Cosine distances + Gaussian kernel with sigma)
2. Data Augmentation via Concatenation (Input-Doubling: pairs (x_i, x_j) with target difference z = y_i - y_j)
3. Random Forest Meta-Classifier & Plurality Voting
4. Hyperparameter Optimization via Dual Annealing / Grid Search
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


class PNNProbabilityExpander:
    """
    First Cascade Stage:
    Computes Euclidean, Chebyshev, and Cosine distance matrices from input vectors to support/training dataset,
    evaluates Gaussian Kernel activations with smoothing parameter sigma,
    and produces intermediate class probability vectors [P_E, P_Cheb, P_Cos].
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = max(sigma, 1e-4)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.idx_class0: Optional[np.ndarray] = None
        self.idx_class1: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PNNProbabilityExpander":
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.idx_class0 = np.where(self.y_train == 0)[0]
        self.idx_class1 = np.where(self.y_train == 1)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates intermediate probability expansion vectors for each sample in X.
        Returns: array of shape (N, 3) containing [P_euclidean, P_chebyshev, P_cosine]
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        eps = 1e-9
        var = 2.0 * (self.sigma ** 2)

        # 1. Euclidean Distance
        d_euclid = cdist(X, self.X_train, metric="euclidean")
        k_euclid = np.exp(- (d_euclid ** 2) / var)

        # 2. Chebyshev Distance
        d_cheb = cdist(X, self.X_train, metric="chebyshev")
        k_cheb = np.exp(- (d_cheb ** 2) / var)

        # 3. Cosine Distance
        d_cos = cdist(X, self.X_train, metric="cosine")
        d_cos = np.nan_to_num(d_cos, nan=1.0)
        k_cos = np.exp(- (d_cos ** 2) / var)

        # Compute posterior probabilities for class 1
        probs = np.zeros((n_samples, 3), dtype=np.float64)
        for m_idx, K_matrix in enumerate([k_euclid, k_cheb, k_cos]):
            sum0 = np.mean(K_matrix[:, self.idx_class0], axis=1) if len(self.idx_class0) > 0 else np.zeros(n_samples)
            sum1 = np.mean(K_matrix[:, self.idx_class1], axis=1) if len(self.idx_class1) > 0 else np.zeros(n_samples)
            probs[:, m_idx] = sum1 / (sum0 + sum1 + eps)

        return probs

    def expand_features(self, X: np.ndarray) -> np.ndarray:
        """Concatenates original features X with intermediate PNN probability outputs."""
        probs = self.transform(X)
        return np.hstack([X, probs])


class CascadeInputDoublingClassifier:
    """
    Cascade-Based Input-Doubling Classifier (Algorithm 1, Izonin et al., 2025).
    Stage 1: PNN probability expansion across distance metrics.
    Stage 2: Input doubling via quadratic pair concatenation and target difference z = y_i - y_j.
    Stage 3: Random Forest meta-classifier.
    Inference: Plurality voting across reference training pairs.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        max_pairs_train: int = 8000,
        n_reference_voting: int = 50,
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
        self.X_train_expanded: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.ref_indices: Optional[np.ndarray] = None
        self.is_fitted = False
        self.tuning_history: List[Dict] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CascadeInputDoublingClassifier":
        """Train the cascade input-doubling classifier."""
        X_train_arr = np.asarray(X_train, dtype=np.float64)
        y_train_arr = np.asarray(y_train, dtype=np.int64)
        n_samples = len(X_train_arr)

        # Stage 1: PNN Probability Expansion
        self.pnn = PNNProbabilityExpander(sigma=self.sigma)
        self.pnn.fit(X_train_arr, y_train_arr)
        self.X_train_expanded = self.pnn.expand_features(X_train_arr)
        self.y_train = y_train_arr

        # Select representative reference samples for voting
        np.random.seed(self.random_state)
        n_ref = min(self.n_reference_voting, n_samples)
        self.ref_indices = np.random.choice(n_samples, size=n_ref, replace=False)

        # Stage 2: Data Augmentation via Concatenation (Input-Doubling)
        # Form pairs (i, j) and compute z = y_i - y_j in {-1, 0, 1}
        pair_i = []
        pair_j = []
        # Stratified pair generation
        for i in range(n_samples):
            # pair with random subset of other samples
            js = np.random.choice([x for x in range(n_samples) if x != i], size=min(n_samples - 1, 55), replace=False)
            for j in js:
                pair_i.append(i)
                pair_j.append(j)

        pair_i = np.array(pair_i)
        pair_j = np.array(pair_j)

        if len(pair_i) > self.max_pairs_train:
            sub_idx = np.random.choice(len(pair_i), size=self.max_pairs_train, replace=False)
            pair_i = pair_i[sub_idx]
            pair_j = pair_j[sub_idx]

        # Concatenate doubled feature vectors [x_exp_i, x_exp_j]
        X_doubled = np.hstack([self.X_train_expanded[pair_i], self.X_train_expanded[pair_j]])
        # Target difference z = y_i - y_j
        z_targets = self.y_train[pair_i] - self.y_train[pair_j]  # values in {-1, 0, 1}

        # Stage 3: Meta-Classifier (Random Forest)
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.rf.fit(X_doubled, z_targets)
        self.is_fitted = True
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs plurality voting across reference training vectors to estimate class probabilities.
        Returns: array of shape (N, 2) with [P(survival=0), P(survival=1)]
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict!")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        n_test = len(X_test_arr)
        X_test_expanded = self.pnn.expand_features(X_test_arr)

        ref_expanded = self.X_train_expanded[self.ref_indices]
        ref_labels = self.y_train[self.ref_indices]
        n_ref = len(ref_labels)

        # For each test sample, construct paired vectors with all reference samples
        # Shape: (n_test * n_ref, 2 * d_expanded)
        test_repeated = np.repeat(X_test_expanded, n_ref, axis=0)
        ref_tiled = np.tile(ref_expanded, (n_test, 1))
        X_doubled_query = np.hstack([test_repeated, ref_tiled])

        # Predict target differences z_hat = y_test - y_ref
        z_preds = self.rf.predict(X_doubled_query)  # shape (n_test * n_ref,)
        z_matrix = z_preds.reshape(n_test, n_ref)

        # Derived test vote: y_test_vote = y_ref + z_hat
        ref_labels_tiled = np.tile(ref_labels, (n_test, 1))
        y_votes = ref_labels_tiled + z_matrix
        # Clip to valid binary range [0, 1]
        y_votes = np.clip(y_votes, 0, 1)

        # Plurality voting: fraction of votes for class 1
        p1 = np.mean(y_votes == 1, axis=1)
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict binary class labels via plurality voting."""
        prob1 = self.predict_proba(X_test)[:, 1]
        return (prob1 >= 0.5).astype(int)

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 15,
        cv_splits: int = 3,
    ) -> Dict:
        """
        Hyperparameter Optimization for sigma, n_estimators, and max_depth
        using Dual Annealing with Stratified Cross-Validation.
        """
        print("[Hyperparameter Optimization] Starting Dual Annealing search...")
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        history: List[Dict] = []
        trial_counter = 0

        def objective(params):
            nonlocal trial_counter
            trial_counter += 1
            sig, n_est, depth = params
            sig = float(sig)
            n_est = int(round(n_est))
            depth = int(round(depth))

            scores = []
            mccs = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]

                clf = CascadeInputDoublingClassifier(
                    sigma=sig,
                    n_estimators=min(n_est, 50),
                    max_depth=depth,
                    max_pairs_train=1200,  # fast during tuning
                    n_reference_voting=20,
                    random_state=42,
                )
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_val)
                scores.append(f1_score(y_val, preds, zero_division=0))
                mccs.append(matthews_corrcoef(y_val, preds))

            mean_f1 = float(np.mean(scores))
            mean_mcc = float(np.mean(mccs))

            history.append({
                "Trial": trial_counter,
                "Sigma": round(sig, 4),
                "Distance_Metrics": "Euclidean, Chebyshev, Cosine",
                "N_Estimators": n_est,
                "Max_Depth": depth,
                "CV_F1_Score": round(mean_f1, 4),
                "CV_MCC": round(mean_mcc, 4),
            })

            # Minimize negative F1 or MCC
            return - (mean_f1 + 0.5 * mean_mcc)

        bounds = [(0.2, 3.5), (50, 150), (4, 14)]
        res = dual_annealing(
            objective,
            bounds=bounds,
            maxiter=n_trials,
            no_local_search=True,
            maxfun=15,
            seed=self.random_state,
        )

        self.tuning_history = history
        best_sigma = float(res.x[0])
        best_n_est = int(round(res.x[1]))
        best_depth = int(round(res.x[2]))

        self.sigma = best_sigma
        self.n_estimators = best_n_est
        self.max_depth = best_depth

        print(f"[Hyperparameter Optimization] Optimal: sigma={best_sigma:.4f}, n_estimators={best_n_est}, max_depth={best_depth}")
        return {
            "optimal_sigma": best_sigma,
            "optimal_n_estimators": best_n_est,
            "optimal_max_depth": best_depth,
            "best_score": -res.fun,
            "history_df": pd.DataFrame(history),
        }
