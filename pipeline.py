"""
pipeline.py - Master Pipeline Execution
Orchestrates:
1. Artificial Data Corruption ("Dirty Data" Step)
2. Data Cleaning & Preprocessing (Imputation, Deduplication, IQR Outlier Capping, Encoding, Scaling)
3. Model 1: Baseline OLS Linear Regression with Bootstrapping (50-100 iterations)
4. Model 2: Cascade-Based Input-Doubling Classifier (PNN + Random Forest, Dual Annealing, Plurality Voting)
5. Comprehensive Performance Benchmarking & Metric Export
6. Visualizations Generation (EDA, Heatmaps, Confusion Matrices, ROC/PR, 3D Scatters)
"""

import os
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.data_corruption import inject_noise_and_corruption
from src.data_cleaning import BoneMarrowPreprocessor
from src.linear_regression import LinearRegressionPipeline
from src.cascade_classifier import CascadeInputDoublingClassifier
from src.evaluation import (
    compute_comprehensive_metrics,
    compute_confusion_matrix_breakdown,
    format_markdown_table,
)
from src.visualizations import (
    plot_missingness_comparison,
    plot_eda_distributions,
    plot_correlation_heatmap,
    plot_confusion_matrices,
    plot_roc_and_pr_curves,
    plot_bootstrapped_regression,
    plot_3d_scatter,
    plot_linear_regression_line_graph,
)


def export_csv(df: pd.DataFrame, root_filename: str):
    """Saves DataFrame to both outputs/ and root directory for convenience."""
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", root_filename)
    df.to_csv(out_path, index=False)
    df.to_csv(root_filename, index=False)
    print(f"[Exported] {out_path} and {root_filename}")


def run_pipeline():
    print("=" * 80)
    print("CASCADED INPUT-DOUBLING ML & LINEAR REGRESSION PIPELINE (Izonin et al., 2025)")
    print("Pediatric Bone Marrow Transplant Survival Analysis")
    print("=" * 80)

    # 1. Load Original Datasets
    train_raw_path = "remaining_dataset.csv"
    test_raw_path = "fully_filled_30.csv"

    if not os.path.exists(train_raw_path) or not os.path.exists(test_raw_path):
        raise FileNotFoundError(f"Missing required datasets: {train_raw_path} or {test_raw_path}")

    df_train_raw = pd.read_csv(train_raw_path)
    df_test_raw = pd.read_csv(test_raw_path)
    print(f"\n[Data Loaded] Training Raw: {df_train_raw.shape}, Test Clean: {df_test_raw.shape}")

    # Copy raw datasets into data/
    os.makedirs("data", exist_ok=True)
    df_train_raw.to_csv("data/remaining_dataset.csv", index=False)
    df_test_raw.to_csv("data/fully_filled_30.csv", index=False)

    # -------------------------------------------------------------------------
    # STEP A: ARTIFICIAL DATA CORRUPTION ("DIRTY DATA")
    # -------------------------------------------------------------------------
    print("\n>>> STEP A: Injecting Artificial Noise & Corruption...")
    df_dirty = inject_noise_and_corruption(
        df_train_raw,
        missing_fraction=0.12,
        num_duplicates=14,
        num_outliers=8,
        random_state=42,
    )
    export_csv(df_dirty, "dirty_dataset.csv")
    df_dirty.to_csv("data/dirty_dataset.csv", index=False)
    print(f"[Corrupted Dataset Created] Shape after corruption: {df_dirty.shape}")

    # -------------------------------------------------------------------------
    # STEP B: DATA CLEANING & PREPROCESSING
    # -------------------------------------------------------------------------
    print("\n>>> STEP B: Cleaning & Preprocessing Datasets...")
    preprocessor = BoneMarrowPreprocessor(iqr_factor=3.0)
    preprocessor.fit(df_dirty)

    df_cleaned_train = preprocessor.transform(df_dirty, is_train=True)
    df_cleaned_test = preprocessor.transform(df_test_raw, is_train=False)

    export_csv(df_cleaned_train, "cleaned_dataset.csv")
    df_cleaned_train.to_csv("data/cleaned_dataset.csv", index=False)
    df_cleaned_test.to_csv("data/cleaned_test_dataset.csv", index=False)
    print(f"[Cleaned Dataset Created] Clean Train Shape: {df_cleaned_train.shape}, Clean Test Shape: {df_cleaned_test.shape}")

    # -------------------------------------------------------------------------
    # VISUALIZATIONS: EDA & CLEANING
    # -------------------------------------------------------------------------
    print("\n>>> Generating Exploratory & Data Cleaning Visualizations...")
    os.makedirs("plots", exist_ok=True)
    plot_missingness_comparison(df_dirty, df_cleaned_train, "plots/missing_values_dirty_vs_clean.png")
    plot_eda_distributions(df_train_raw, "plots/eda_feature_distributions.png")
    plot_correlation_heatmap(df_cleaned_train, "plots/eda_correlation_heatmap.png")
    plot_3d_scatter(df_train_raw, "Donorage", "Recipientage", "CD34kgx10d6", "plots/3d_scatter_features.png")

    # Prepare Feature Matrices
    feature_cols = [c for c in df_cleaned_train.columns if c not in ["id", "survival_status"]]
    X_train = df_cleaned_train[feature_cols]
    y_train = df_cleaned_train["survival_status"]

    X_test = df_cleaned_test[feature_cols]
    y_test = df_cleaned_test["survival_status"].values
    test_ids = df_cleaned_test["id"] if "id" in df_cleaned_test.columns else pd.Series(range(1, len(X_test) + 1))

    # -------------------------------------------------------------------------
    # MODEL 1: LINEAR REGRESSION WITH BOOTSTRAPPING
    # -------------------------------------------------------------------------
    print("\n>>> Training Model 1: Ordinary Least Squares (OLS) with Bootstrapping...")
    t0_lr = time.time()
    lr_pipe = LinearRegressionPipeline(n_bootstrap=100, random_state=42)
    lr_pipe.fit(X_train, y_train)
    lr_time = time.time() - t0_lr

    y_lr_cont, y_lr_bin = lr_pipe.predict(X_test)
    lr_metrics = compute_comprehensive_metrics("Linear_Regression_OLS", y_test, y_lr_bin, y_lr_cont, lr_time)
    lr_residuals_df = lr_pipe.compute_residuals_table(test_ids, y_test, y_lr_cont)

    export_csv(lr_pipe.bootstrap_coefs_df, "bootstrapped_coefficients.csv")
    export_csv(lr_residuals_df, "regression_errors.csv")
    print(f"[Model 1 Evaluated] Accuracy: {lr_metrics['Accuracy']}, F1: {lr_metrics['F1_Score']}, RMSE: {lr_metrics['RMSE']}")

    # -------------------------------------------------------------------------
    # MODEL 2: CASCADE-BASED INPUT-DOUBLING CLASSIFIER (Izonin et al., 2025)
    # -------------------------------------------------------------------------
    print("\n>>> Training Model 2: Cascade-Based Input-Doubling Classifier (PNN + Random Forest)...")
    cascade_clf = CascadeInputDoublingClassifier(
        sigma=1.2,
        n_estimators=100,
        max_depth=8,
        max_pairs_train=8000,
        n_reference_voting=45,
        random_state=42,
    )

    # Hyperparameter Optimization via Dual Annealing
    t0_opt = time.time()
    opt_res = cascade_clf.optimize_hyperparameters(X_train.values, y_train.values, n_trials=10, cv_splits=3)
    export_csv(opt_res["history_df"], "hyperparameter_tuning_results.csv")

    # Fit final cascade model with optimal parameters
    t0_cas = time.time()
    cascade_clf.fit(X_train.values, y_train.values)
    cas_time = time.time() - t0_cas

    # Inference via Plurality Voting
    y_cas_probs = cascade_clf.predict_proba(X_test.values)[:, 1]
    y_cas_bin = cascade_clf.predict(X_test.values)
    cas_metrics = compute_comprehensive_metrics("Hybrid_Cascade_Input_Doubling", y_test, y_cas_bin, y_cas_probs, cas_time)
    print(f"[Model 2 Evaluated] Accuracy: {cas_metrics['Accuracy']}, F1: {cas_metrics['F1_Score']}, MCC: {cas_metrics['MCC']}")

    # Extract Feature Importance from Random Forest Meta-Classifier
    rf_feat_imp = cascade_clf.rf.feature_importances_
    # Doubled feature size: 2 * (n_features + 3 PNN probs)
    half_size = len(rf_feat_imp) // 2
    doubled_names = feature_cols + ["PNN_Euclid", "PNN_Chebyshev", "PNN_Cosine"]
    aggregated_imp = rf_feat_imp[:half_size] + rf_feat_imp[half_size:]
    feat_imp_df = pd.DataFrame({
        "Feature": doubled_names,
        "Importance_CascadeRF": np.round(aggregated_imp / np.sum(aggregated_imp), 5),
    }).sort_values(by="Importance_CascadeRF", ascending=False)
    export_csv(feat_imp_df, "feature_importance.csv")

    # -------------------------------------------------------------------------
    # GENERATE DELIVERABLE CSV FILES
    # -------------------------------------------------------------------------
    print("\n>>> Generating Deliverable CSV Files & Ensemble Blend...")

    # Model 3: Combined Ensemble Blend (Both Models Together)
    y_ens_prob = 0.5 * np.clip(y_lr_cont, 0, 1) + 0.5 * y_cas_probs
    y_ens_bin = (y_ens_prob >= 0.5).astype(int)
    ens_metrics = compute_comprehensive_metrics("Combined_Ensemble_Blend", y_test, y_ens_bin, y_ens_prob, lr_time + cas_time)

    # 1. final_predictions.csv
    match_status = np.where(
        (y_lr_bin == y_cas_bin) & (y_cas_bin == y_ens_bin),
        "All_Models_Agree",
        np.where(y_lr_bin == y_cas_bin, "Both_Individual_Agree", "Disagreement")
    )
    final_pred_df = pd.DataFrame({
        "Sample_ID": test_ids.values,
        "Actual_Survival": y_test,
        "Predicted_Survival_LinearRegression": y_lr_bin,
        "Predicted_Survival_HybridCascade": y_cas_bin,
        "Predicted_Survival_CombinedEnsemble": y_ens_bin,
        "LinearRegression_Continuous": np.round(y_lr_cont, 4),
        "HybridCascade_Probability": np.round(y_cas_probs, 4),
        "CombinedEnsemble_Probability": np.round(y_ens_prob, 4),
        "Match_Status": match_status,
    })
    export_csv(final_pred_df, "final_predictions.csv")

    # 2. model_metrics.csv
    metrics_summary_df = pd.DataFrame([lr_metrics, cas_metrics, ens_metrics])
    export_csv(metrics_summary_df, "model_metrics.csv")

    # 3. confusion_matrix_metrics.csv
    cm_lr = compute_confusion_matrix_breakdown("Linear_Regression_OLS", y_test, y_lr_bin)
    cm_cas = compute_confusion_matrix_breakdown("Hybrid_Cascade_Input_Doubling", y_test, y_cas_bin)
    cm_ens = compute_confusion_matrix_breakdown("Combined_Ensemble_Blend", y_test, y_ens_bin)
    cm_summary_df = pd.DataFrame([cm_lr, cm_cas, cm_ens])
    export_csv(cm_summary_df, "confusion_matrix_metrics.csv")

    # 4. model_accuracy_and_error_reduction.csv
    base_mae = lr_metrics["MAE"]
    base_rmse = lr_metrics["RMSE"]
    base_acc = lr_metrics["Accuracy"]
    reduc_rows = []
    for m in [lr_metrics, cas_metrics, ens_metrics]:
        mae_reduc_pct = ((base_mae - m["MAE"]) / base_mae) * 100.0
        rmse_reduc_pct = ((base_rmse - m["RMSE"]) / base_rmse) * 100.0
        acc_impr_pct = ((m["Accuracy"] - base_acc) / base_acc) * 100.0
        reduc_rows.append({
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
    reduc_df = pd.DataFrame(reduc_rows)
    export_csv(reduc_df, "model_accuracy_and_error_reduction.csv")

    # -------------------------------------------------------------------------
    # GENERATE EVALUATION PLOTS
    # -------------------------------------------------------------------------
    print("\n>>> Generating Model Evaluation Plots...")
    plot_confusion_matrices(y_test, y_lr_bin, y_cas_bin, "plots/confusion_matrix_comparison.png")
    plot_roc_and_pr_curves(y_test, y_lr_cont, y_cas_probs, "plots/roc_pr_curves.png")
    plot_linear_regression_line_graph(
        y_test,
        y_lr_cont,
        output_path="plots/linear_regression_survival_status_line_graph.png",
        x_label="Actual Clinical Survival Outcome (0=Survived, 1=Deceased)",
        y_label="Model 1 OLS Predicted Continuous Score",
        title="Model 1: OLS Linear Regression Line Graph (Outcome Prediction)",
        unit="Score",
    )
    if "survival_time" in X_test.columns:
        plot_bootstrapped_regression(X_test["survival_time"].values, y_test, lr_pipe, "survival_time", "plots/linear_regression_bootstrapped_fits.png")
    elif "Donorage" in X_test.columns:
        plot_bootstrapped_regression(X_test["Donorage"].values, y_test, lr_pipe, "Donorage", "plots/linear_regression_bootstrapped_fits.png")

    # -------------------------------------------------------------------------
    # PRINT STRUCTURED MARKDOWN BENCHMARK TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK RESULTS (Matching Paper Format)")
    print("=" * 80)

    print(format_markdown_table(metrics_summary_df, "Comprehensive Model Performance Comparison Table"))
    print(format_markdown_table(cm_summary_df, "Confusion Matrix Breakdown & Diagnostic Metrics Table"))

    tuning_table = pd.DataFrame([{
        "Parameter": "Smooth factor (sigma)",
        "Optimal_Value": f"{opt_res['optimal_sigma']:.4f}",
        "Search_Method": "Dual Annealing Global Optimization",
    }, {
        "Parameter": "Distance Metrics",
        "Optimal_Value": "Euclidean, Chebyshev, Cosine",
        "Search_Method": "PNN Probability Expansion Layer",
    }, {
        "Parameter": "Number of Estimators (n_estimators)",
        "Optimal_Value": str(opt_res["optimal_n_estimators"]),
        "Search_Method": "Dual Annealing Global Optimization",
    }, {
        "Parameter": "Maximum Tree Depth (max_depth)",
        "Optimal_Value": str(opt_res["optimal_max_depth"]),
        "Search_Method": "Dual Annealing Global Optimization",
    }, {
        "Parameter": "Ensemble Voting Method",
        "Optimal_Value": "Plurality Voting across Doubled Vector Pairs",
        "Search_Method": "Algorithm 1 Framework",
    }])
    print(format_markdown_table(tuning_table, "Parameter Optimization Results Table"))

    print("\n>>> Pipeline execution completed successfully! All files generated.\n")


if __name__ == "__main__":
    run_pipeline()
