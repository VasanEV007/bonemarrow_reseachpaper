"""
train_and_export.py
Trains both Model 1 (Linear Regression for continuous survival_time)
and Model 2 (PNN + Random Forest Cascade Classifier for binary survival_status)
using the 35 verified patient features, and exports serialized artifacts,
metrics, and evaluation chart data to bmt_webapp/models/.
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

# Ensure bmt_webapp is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bmt_webapp.backend.models_core import (
    CascadeClassifierAppModel,
    ClinicalBMTPreprocessor,
    PNNProbabilityExpander,
)


def train_and_export():
    print("=" * 70)
    print("TRAINING & EXPORTING MODELS FOR BMT SURVIVAL PREDICTION WEB APP")
    print("=" * 70)

    models_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load raw datasets from project root
    train_csv = os.path.join(project_root, "remaining_dataset.csv")
    test_csv = os.path.join(project_root, "fully_filled_30.csv")

    df_train_raw = pd.read_csv(train_csv)
    df_test_raw = pd.read_csv(test_csv)

    print(f"Loaded Raw Train: {df_train_raw.shape}, Raw Test: {df_test_raw.shape}")

    # 1. Fit Preprocessor
    preprocessor = ClinicalBMTPreprocessor(iqr_factor=3.0)
    preprocessor.fit(df_train_raw)

    X_train_df = preprocessor.transform(df_train_raw)
    X_test_df = preprocessor.transform(df_test_raw)

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Targets
    y_train_time = pd.to_numeric(df_train_raw["survival_time"], errors="coerce").fillna(df_train_raw["survival_time"].median()).values
    y_test_time = pd.to_numeric(df_test_raw["survival_time"], errors="coerce").values

    y_train_status = pd.to_numeric(df_train_raw["survival_status"], errors="coerce").fillna(0).astype(int).values
    y_test_status = pd.to_numeric(df_test_raw["survival_status"], errors="coerce").fillna(0).astype(int).values

    print(f"Features dimension after preprocessing: {X_train.shape[1]}")

    # -------------------------------------------------------------
    # MODEL 1: LINEAR REGRESSION (Predicting continuous survival_time)
    # -------------------------------------------------------------
    print("\n[Training Model 1] OLS Linear Regression for survival_time (days)...")
    t0_lr = time.time()
    lr_model = Ridge(alpha=1.0, random_state=42)
    lr_model.fit(X_train, y_train_time)
    lr_time = time.time() - t0_lr

    y_pred_time = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test_time, y_pred_time))
    mae_lr = mean_absolute_error(y_test_time, y_pred_time)
    r2_lr = r2_score(y_test_time, y_pred_time)
    residuals_lr = y_test_time - y_pred_time

    # 100-iteration bootstrapping for parameter confidence intervals
    boot_coefs = []
    np.random.seed(42)
    for _ in range(100):
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        m = Ridge(alpha=1.0, random_state=42).fit(X_train[idx], y_train_time[idx])
        boot_coefs.append(m.coef_)
    boot_coefs = np.array(boot_coefs)
    coef_means = np.mean(boot_coefs, axis=0)
    coef_ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    coef_ci_high = np.percentile(boot_coefs, 97.5, axis=0)

    lr_feature_importance = [
        {
            "feature": name,
            "coefficient": round(float(coef), 2),
            "ci_lower_95": round(float(low), 2),
            "ci_upper_95": round(float(high), 2),
        }
        for name, coef, low, high in zip(preprocessor.transformed_feature_names, coef_means, coef_ci_low, coef_ci_high)
    ]
    lr_feature_importance = sorted(lr_feature_importance, key=lambda x: abs(x["coefficient"]), reverse=True)

    print(f"Model 1 (Regression): RMSE = {rmse_lr:.2f} days, MAE = {mae_lr:.2f} days, R² = {r2_lr:.4f}")

    # -------------------------------------------------------------
    # MODEL 2: HYBRID CASCADE CLASSIFIER (Predicting survival_status)
    # -------------------------------------------------------------
    print("\n[Training Model 2] PNN + Random Forest Cascade Classifier for survival_status...")
    t0_cas = time.time()
    cascade_model = CascadeClassifierAppModel(
        sigma=3.2183,
        n_estimators=100,
        max_depth=9,
        max_pairs_train=6000,
        n_reference_voting=40,
        random_state=42,
    )
    cascade_model.fit(X_train, y_train_status)
    cas_time = time.time() - t0_cas

    y_pred_cas_prob = cascade_model.predict_proba(X_test)[:, 1]
    y_pred_cas_bin = cascade_model.predict(X_test)

    acc_cas = accuracy_score(y_test_status, y_pred_cas_bin)
    prec_cas = precision_score(y_test_status, y_pred_cas_bin, zero_division=0)
    rec_cas = recall_score(y_test_status, y_pred_cas_bin, zero_division=0)
    f1_cas = f1_score(y_test_status, y_pred_cas_bin, zero_division=0)
    mcc_cas = matthews_corrcoef(y_test_status, y_pred_cas_bin)
    kappa_cas = cohen_kappa_score(y_test_status, y_pred_cas_bin)

    cm_cas = confusion_matrix(y_test_status, y_pred_cas_bin)
    tn, fp, fn, tp = cm_cas.ravel()

    fpr, tpr, _ = roc_curve(y_test_status, y_pred_cas_prob)
    roc_auc_cas = auc(fpr, tpr)
    p_curve, r_curve, _ = precision_recall_curve(y_test_status, y_pred_cas_prob)
    pr_auc_cas = auc(r_curve, p_curve)

    rf_imps = cascade_model.rf.feature_importances_
    half = len(rf_imps) // 2
    doubled_names = preprocessor.transformed_feature_names + ["PNN_Euclidean", "PNN_Chebyshev", "PNN_Cosine"]
    agg_imp = rf_imps[:half] + rf_imps[half:]
    agg_imp = agg_imp / np.sum(agg_imp)

    cas_feature_importance = [
        {"feature": name, "importance": round(float(val), 4)}
        for name, val in sorted(zip(doubled_names, agg_imp), key=lambda x: x[1], reverse=True)
    ]

    print(f"Model 2 (Classification): Accuracy = {acc_cas:.4f}, Precision = {prec_cas:.4f}, Recall = {rec_cas:.4f}, F1 = {f1_cas:.4f}, MCC = {mcc_cas:.4f}")

    # -------------------------------------------------------------
    # SERIALIZE ARTIFACTS
    # -------------------------------------------------------------
    print("\n[Serializing Models with Joblib]...")
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(lr_model, os.path.join(models_dir, "linear_regression_survival_time.joblib"))
    joblib.dump(cascade_model, os.path.join(models_dir, "cascade_classifier_survival_status.joblib"))

    metrics_data = {
        "model_1_regression": {
            "name": "Linear Regression (Ridge Regularized OLS)",
            "type": "Regression",
            "target": "survival_time",
            "target_units": "days",
            "rmse": round(float(rmse_lr), 2),
            "mae": round(float(mae_lr), 2),
            "r2_score": round(float(r2_lr), 4),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_sec": round(float(lr_time), 3),
            "top_features": lr_feature_importance[:10],
        },
        "model_2_classification": {
            "name": "PNN + Random Forest Cascade-Based Input-Doubling Classifier",
            "short_name": "Hybrid Cascade Classifier (Algorithm 1)",
            "research_reference": "Izonin et al. (2025), Computation, MDPI. DOI: 10.3390/computation13040080",
            "type": "Binary Classification",
            "target": "survival_status",
            "target_classes": {"0": "Survived", "1": "Deceased"},
            "accuracy": round(float(acc_cas), 4),
            "precision": round(float(prec_cas), 4),
            "recall": round(float(rec_cas), 4),
            "f1_score": round(float(f1_cas), 4),
            "mcc": round(float(mcc_cas), 4),
            "cohens_kappa": round(float(kappa_cas), 4),
            "roc_auc": round(float(roc_auc_cas), 4),
            "pr_auc": round(float(pr_auc_cas), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
                "sensitivity": round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0,
            },
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_sec": round(float(cas_time), 3),
            "top_features": cas_feature_importance[:10],
        },
    }

    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    chart_data = {
        "regression_actual_vs_predicted": {
            "sample_ids": [int(x) for x in df_test_raw["id"].values],
            "actual_time": [round(float(v), 1) for v in y_test_time],
            "predicted_time": [round(float(v), 1) for v in y_pred_time],
            "residuals": [round(float(v), 1) for v in residuals_lr],
        },
        "classification_roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
            "auc": round(float(roc_auc_cas), 4),
        },
        "classification_pr_curve": {
            "recall": [round(float(v), 4) for v in r_curve],
            "precision": [round(float(v), 4) for v in p_curve],
            "auc": round(float(pr_auc_cas), 4),
        },
        "test_classification_results": {
            "sample_ids": [int(x) for x in df_test_raw["id"].values],
            "actual_class": [int(x) for x in y_test_status],
            "predicted_class": [int(x) for x in y_pred_cas_bin],
            "predicted_probability": [round(float(v), 4) for v in y_pred_cas_prob],
        },
        "feature_distributions": {
            "donorage": [round(float(v), 1) for v in df_train_raw["Donorage"].dropna().values],
            "recipientage": [round(float(v), 1) for v in df_train_raw["Recipientage"].dropna().values],
            "cd34": [round(float(v), 2) for v in df_train_raw["CD34kgx10d6"].dropna().values],
            "survival_time": [round(float(v), 0) for v in y_train_time],
            "disease_counts": df_train_raw["Disease"].value_counts().to_dict(),
            "status_counts": {"Survived (0)": int(np.sum(y_train_status == 0)), "Deceased (1)": int(np.sum(y_train_status == 1))},
        },
        "feature_importances": {
            "regression_top": lr_feature_importance[:12],
            "cascade_top": cas_feature_importance[:12],
        },
    }

    with open(os.path.join(models_dir, "evaluation_charts_data.json"), "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=2)

    print("\n[Successfully Exported All Model Artifacts & Metrics!]")


if __name__ == "__main__":
    train_and_export()
