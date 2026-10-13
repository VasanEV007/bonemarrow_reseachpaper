"""
generate_error_reduction_report.py
Generates model_accuracy_and_error_reduction.csv containing:
- Accuracy, Precision, Recall, F1, MCC
- RMSE, MAE
- Error Reduction Percentages (MAE and RMSE reduction vs Baseline)
- Accuracy Improvement Percentages
For:
1. Linear Regression OLS (Baseline)
2. Hybrid Cascade Input-Doubling (Algorithm 1)
3. Combined Ensemble Blend (Both Models Together)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.data_cleaning import BoneMarrowPreprocessor
from src.linear_regression import LinearRegressionPipeline
from src.cascade_classifier import CascadeInputDoublingClassifier


def main():
    print("=" * 80)
    print("GENERATING MODEL ACCURACY & ERROR REDUCTION REPORT")
    print("=" * 80)

    # 1. Load data
    df_dirty = pd.read_csv("dirty_dataset.csv")
    df_test = pd.read_csv("fully_filled_30.csv")

    preprocessor = BoneMarrowPreprocessor(iqr_factor=3.0).fit(df_dirty)
    df_train_cl = preprocessor.transform(df_dirty, is_train=True)
    df_test_cl = preprocessor.transform(df_test, is_train=False)

    feature_cols = [c for c in df_train_cl.columns if c not in ["id", "survival_status"]]
    X_train = df_train_cl[feature_cols].values
    y_train = df_train_cl["survival_status"].values
    X_test = df_test_cl[feature_cols].values
    y_test = df_test_cl["survival_status"].values
    sample_ids = df_test_cl["id"].values

    # 2. Model 1: Linear Regression
    lr = LinearRegressionPipeline(n_bootstrap=100, random_state=42)
    lr.fit(df_train_cl[feature_cols], df_train_cl["survival_status"])
    y_lr_cont, y_lr_bin = lr.predict(df_test_cl[feature_cols])

    # 3. Model 2: Hybrid Cascade
    cas = CascadeInputDoublingClassifier(
        sigma=3.2183,
        n_estimators=80,
        max_depth=9,
        max_pairs_train=5000,
        n_reference_voting=35,
        random_state=42,
    )
    cas.fit(X_train, y_train)
    y_cas_prob = cas.predict_proba(X_test)[:, 1]
    y_cas_bin = cas.predict(X_test)

    # 4. Model 3: Combined / Ensemble (Both Models Together)
    # Balanced soft blend between linear probability and cascade neural-tree density
    y_ens_prob = 0.5 * np.clip(y_lr_cont, 0, 1) + 0.5 * y_cas_prob
    y_ens_bin = (y_ens_prob >= 0.5).astype(int)

    # Evaluate each
    def get_metrics(name, y_true, y_pred_bin, y_scores):
        acc = accuracy_score(y_true, y_pred_bin)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred_bin)
        kappa = cohen_kappa_score(y_true, y_pred_bin)
        mae = mean_absolute_error(y_true, y_scores)
        rmse = np.sqrt(mean_squared_error(y_true, y_scores))
        return {
            "Model_Name": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "MCC": mcc,
            "Cohens_Kappa": kappa,
            "MAE": mae,
            "RMSE": rmse,
        }

    m_lr = get_metrics("Linear_Regression_OLS (Baseline)", y_test, y_lr_bin, y_lr_cont)
    m_cas = get_metrics("Hybrid_Cascade_Input_Doubling (Algorithm 1)", y_test, y_cas_bin, y_cas_prob)
    m_ens = get_metrics("Combined_Ensemble_Blend (Both Models Together)", y_test, y_ens_bin, y_ens_prob)

    models_data = [m_lr, m_cas, m_ens]

    # Baseline references (Linear Regression)
    base_mae = m_lr["MAE"]
    base_rmse = m_lr["RMSE"]
    base_acc = m_lr["Accuracy"]

    rows = []
    for m in models_data:
        mae_reduc_pct = ((base_mae - m["MAE"]) / base_mae) * 100.0
        rmse_reduc_pct = ((base_rmse - m["RMSE"]) / base_rmse) * 100.0
        acc_impr_pct = ((m["Accuracy"] - base_acc) / base_acc) * 100.0

        rows.append({
            "Model_Name": m["Model_Name"],
            "Accuracy": round(m["Accuracy"], 4),
            "Accuracy_Improvement_Percent": f"{acc_impr_pct:+.2f}%",
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1_Score": round(m["F1_Score"], 4),
            "MCC": round(m["MCC"], 4),
            "Cohens_Kappa": round(m["Cohens_Kappa"], 4),
            "RMSE": round(m["RMSE"], 4),
            "RMSE_Error_Reduction_Percent": f"{rmse_reduc_pct:+.2f}%",
            "MAE": round(m["MAE"], 4),
            "MAE_Error_Reduction_Percent": f"{mae_reduc_pct:+.2f}%",
        })

    report_df = pd.DataFrame(rows)

    # Export to root and outputs/
    report_df.to_csv("model_accuracy_and_error_reduction.csv", index=False)
    os.makedirs("outputs", exist_ok=True)
    report_df.to_csv("outputs/model_accuracy_and_error_reduction.csv", index=False)
    print("\n[Exported] model_accuracy_and_error_reduction.csv")

    # Update final_predictions.csv to include Both Models Together
    final_preds_df = pd.DataFrame({
        "Sample_ID": sample_ids,
        "Actual_Survival": y_test,
        "Predicted_Survival_LinearRegression": y_lr_bin,
        "Predicted_Survival_HybridCascade": y_cas_bin,
        "Predicted_Survival_CombinedEnsemble": y_ens_bin,
        "LinearRegression_Continuous": np.round(y_lr_cont, 4),
        "HybridCascade_Probability": np.round(y_cas_prob, 4),
        "CombinedEnsemble_Probability": np.round(y_ens_prob, 4),
        "Match_Status": np.where(
            (y_lr_bin == y_cas_bin) & (y_cas_bin == y_ens_bin),
            "All_Models_Agree",
            np.where(y_lr_bin == y_cas_bin, "Both_Individual_Agree", "Disagreement")
        ),
    })
    final_preds_df.to_csv("final_predictions.csv", index=False)
    final_preds_df.to_csv("outputs/final_predictions.csv", index=False)
    print("[Updated] final_predictions.csv (now includes individual and combined ensemble predictions)")

    print("\n" + "=" * 80)
    print("MODEL ACCURACY & ERROR REDUCTION REPORT TABLE")
    print("=" * 80)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
