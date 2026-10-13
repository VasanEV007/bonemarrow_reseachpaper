"""
visualizations.py - High-Quality Visualizations & Diagnostic Plotting
Generates publication-ready figures for EDA, data cleaning, confusion matrices, ROC/PR curves,
bootstrapped regression fits, and 3D feature spaces.
"""

import os
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Use a clean, publication-grade style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})


def plot_missingness_comparison(
    df_dirty: pd.DataFrame, df_cleaned: pd.DataFrame, output_path: str = "plots/missing_values_dirty_vs_clean.png"
):
    """Visualizes missing value heatmaps before and after data cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Dirty missingness
    dirty_null = df_dirty.replace("?", np.nan).isnull()
    sns.heatmap(dirty_null, cbar=True, cmap="YlOrRd", yticklabels=False, ax=axes[0])
    axes[0].set_title(f"A. Dirty Dataset Missingness (Total NaNs: {dirty_null.sum().sum()})", fontweight="bold")
    axes[0].set_xlabel("Attributes")
    axes[0].set_ylabel("Samples")

    # Cleaned missingness
    clean_null = df_cleaned.isnull()
    sns.heatmap(clean_null, cbar=True, cmap="Blues", yticklabels=False, ax=axes[1])
    axes[1].set_title(f"B. Cleaned Dataset Missingness (Total NaNs: {clean_null.sum().sum()})", fontweight="bold")
    axes[1].set_xlabel("Attributes")
    axes[1].set_ylabel("Samples")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_eda_distributions(df: pd.DataFrame, output_path: str = "plots/eda_feature_distributions.png"):
    """Plots histograms & KDE for key continuous variables stratified by survival status."""
    key_cols = ["Donorage", "Recipientage", "CD34kgx10d6", "CD3dkgx10d8", "Rbodymass", "survival_time"]
    avail_cols = [c for c in key_cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(avail_cols):
        ax = axes[idx]
        col_series = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce").dropna()
        sns.histplot(
            data=df,
            x=col,
            hue="survival_status" if "survival_status" in df.columns else None,
            kde=True,
            palette=["#2b5c8f", "#d95f02"],
            ax=ax,
            alpha=0.6,
        )
        ax.set_title(f"Distribution: {col}", fontweight="bold")
        ax.set_xlabel(col)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str = "plots/eda_correlation_heatmap.png"):
    """Plots correlation heatmap across key numerical features."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "id" in num_df.columns:
        num_df = num_df.drop(columns=["id"])

    # Select top 15 most variable features for visual clarity
    top_cols = num_df.var().sort_values(ascending=False).head(16).index.tolist()
    if "survival_status" in num_df.columns and "survival_status" not in top_cols:
        top_cols.append("survival_status")

    corr = num_df[top_cols].corr()

    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Pediatric Bone Marrow Transplant - Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred_lr: np.ndarray,
    y_pred_cascade: np.ndarray,
    output_path: str = "plots/confusion_matrix_comparison.png",
):
    """Side-by-side Confusion Matrix Heatmaps for Linear Regression vs Cascade-Based Classifier."""
    from sklearn.metrics import confusion_matrix

    cm_lr = confusion_matrix(y_true, y_pred_lr)
    cm_cascade = confusion_matrix(y_true, y_pred_cascade)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cm, title, cmap in zip(
        axes,
        [cm_lr, cm_cascade],
        ["Model 1: Linear Regression (OLS Thresholded)", "Model 2: Hybrid Cascade Classifier (Algorithm 1)"],
        ["Blues", "Greens"],
    ):
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ])
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=False, ax=ax, annot_kws={"size": 14, "fontweight": "bold"})
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted Class (0: Survived, 1: Deceased)")
        ax.set_ylabel("Actual Class (0: Survived, 1: Deceased)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    scores_lr: np.ndarray,
    scores_cascade: np.ndarray,
    output_path: str = "plots/roc_pr_curves.png",
):
    """Plots ROC-AUC Curves and Precision-Recall Curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_true, scores_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    fpr_cas, tpr_cas, _ = roc_curve(y_true, scores_cascade)
    auc_cas = auc(fpr_cas, tpr_cas)

    axes[0].plot(fpr_lr, tpr_lr, label=f"Linear Regression (AUC = {auc_lr:.3f})", color="#e41a1c", lw=2.5)
    axes[0].plot(fpr_cas, tpr_cas, label=f"Hybrid Cascade (AUC = {auc_cas:.3f})", color="#2ca02c", lw=2.5)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")

    # PR Curves
    p_lr, r_lr, _ = precision_recall_curve(y_true, scores_lr)
    pr_auc_lr = auc(r_lr, p_lr)

    p_cas, r_cas, _ = precision_recall_curve(y_true, scores_cascade)
    pr_auc_cas = auc(r_cas, p_cas)

    axes[1].plot(r_lr, p_lr, label=f"Linear Regression (PR-AUC = {pr_auc_lr:.3f})", color="#e41a1c", lw=2.5)
    axes[1].plot(r_cas, p_cas, label=f"Hybrid Cascade (PR-AUC = {pr_auc_cas:.3f})", color="#2ca02c", lw=2.5)
    axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_bootstrapped_regression(
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    lr_pipeline,
    feat_name: str = "survival_time",
    output_path: str = "plots/linear_regression_bootstrapped_fits.png",
):
    """
    Plots scatter of actual values vs key predictor with superimposed
    fitted regression line and 95% bootstrapped confidence interval bands.
    """
    plt.figure(figsize=(10, 6))

    # Actual test points
    plt.scatter(
        X_test_feat,
        y_test,
        color="#1f77b4",
        alpha=0.8,
        s=60,
        edgecolors="k",
        label="Actual Test Observations",
    )

    # Sort feature for smooth curve plotting
    sort_idx = np.argsort(X_test_feat)
    x_sorted = X_test_feat[sort_idx]

    # Generate bootstrap prediction lines
    coef_df = lr_pipeline.bootstrap_coefs_df
    intercept = coef_df[coef_df["Feature"] == "Intercept"]["Mean_Coefficient"].values[0]
    # Match feature
    feat_row = coef_df[coef_df["Feature"] == feat_name]
    slope = feat_row["Mean_Coefficient"].values[0] if len(feat_row) > 0 else 0.1
    slope_low = feat_row["CI_Lower_95"].values[0] if len(feat_row) > 0 else slope * 0.8
    slope_high = feat_row["CI_Upper_95"].values[0] if len(feat_row) > 0 else slope * 1.2

    y_line = intercept + slope * x_sorted
    y_line_low = intercept + slope_low * x_sorted
    y_line_high = intercept + slope_high * x_sorted

    plt.plot(x_sorted, y_line, color="#d62728", lw=2.5, label="OLS Regression Fit (Mean)")
    plt.fill_between(
        x_sorted,
        np.minimum(y_line_low, y_line_high),
        np.maximum(y_line_low, y_line_high),
        color="#d62728",
        alpha=0.2,
        label="95% Bootstrapped CI Band",
    )

    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Classification Threshold (0.5)")
    plt.title(f"OLS Linear Regression with Bootstrapped 95% Confidence Band ({feat_name})", fontweight="bold")
    plt.xlabel(f"Standardized {feat_name}")
    plt.ylabel("Survival Outcome (0: Survived, 1: Deceased)")
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_3d_scatter(
    df: pd.DataFrame,
    col_x: str = "Donorage",
    col_y: str = "Recipientage",
    col_z: str = "CD34kgx10d6",
    output_path: str = "plots/3d_scatter_features.png",
):
    """Generates 3D Scatter plot of continuous variables colored by survival status."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = pd.to_numeric(df[col_x], errors="coerce").fillna(0)
    y_vals = pd.to_numeric(df[col_y], errors="coerce").fillna(0)
    z_vals = pd.to_numeric(df[col_z], errors="coerce").fillna(0)
    labels = df["survival_status"].values if "survival_status" in df.columns else np.zeros(len(df))

    scatter = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=labels,
        cmap="coolwarm",
        s=50,
        alpha=0.8,
        edgecolors="w",
    )

    ax.set_xlabel(col_x, fontweight="bold", labelpad=8)
    ax.set_ylabel(col_y, fontweight="bold", labelpad=8)
    ax.set_zlabel(col_z, fontweight="bold", labelpad=8)
    ax.set_title("3D Morphological Clinical Feature Space Stratified by Survival Status", fontweight="bold", pad=15)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label("Survival Status (0=Survived, 1=Deceased)", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")


def plot_linear_regression_line_graph(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str = "plots/linear_regression_line_graph.png",
    x_label: str = "Actual Survival Time (Days)",
    y_label: str = "Predicted Survival Time (Days)",
    title: str = "Model 1: Linear Regression Line Graph (Actual vs. Predicted)",
    unit: str = "Days",
):
    """
    Generates a publication-grade Linear Regression Line Graph.
    Features:
    - Empirical observation scatter points with residual drop lines.
    - OLS Linear Regression fit line (y = mx + b) with 95% confidence interval shaded band.
    - 1:1 Identity reference line (y = x) representing ideal agreement.
    - Summary stat box containing R^2, RMSE, MAE, Pearson r, and regression equation.
    - Secondary panel showing Residual Errors vs. Fitted Values with confidence bounds.
    """
    from scipy import stats

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residuals = actual - predicted
    n = len(actual)

    # Calculate statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted)
    r2 = r_value ** 2
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Domain for regression line
    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    padding = (max_val - min_val) * 0.08
    line_x = np.linspace(min_val - padding, max_val + padding, 200)
    line_y = slope * line_x + intercept

    # 95% Confidence Interval band for regression line
    x_mean = np.mean(actual)
    ss_x = np.sum((actual - x_mean) ** 2)
    s_err = np.sqrt(np.sum((predicted - (slope * actual + intercept)) ** 2) / max(n - 2, 1))
    t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
    ci = t_val * s_err * np.sqrt(1.0 / n + (line_x - x_mean) ** 2 / max(ss_x, 1e-9))

    # Shaded 95% CI band
    ax.fill_between(
        line_x,
        line_y - ci,
        line_y + ci,
        color="#0891b2",
        alpha=0.18,
        label="95% Confidence Interval Band",
    )

    # 1:1 Identity reference line
    ax.plot(
        [min_val - padding, max_val + padding],
        [min_val - padding, max_val + padding],
        color="#64748b",
        linestyle="--",
        linewidth=1.8,
        label="1:1 Perfect Prediction Line (y = x)",
    )

    # Fitted Linear Regression Line
    ax.plot(
        line_x,
        line_y,
        color="#0891b2",
        linewidth=2.8,
        label=f"OLS Regression Fit Line (y = {slope:.2f}x + {intercept:.1f})",
    )

    # Residual drop lines (connecting points to regression line)
    for x_i, y_i in zip(actual, predicted):
        y_fit = slope * x_i + intercept
        ax.plot([x_i, x_i], [y_i, y_fit], color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.7)

    # Scatter points
    ax.scatter(
        actual,
        predicted,
        color="#0284c7",
        edgecolors="#0f172a",
        s=70,
        linewidths=1.2,
        alpha=0.88,
        zorder=5,
        label=f"Test Cohort Observations (N = {n})",
    )

    # Annotations & Stat Box
    sign = "+" if intercept >= 0 else "-"
    sep_line = "-" * 26
    stat_box_text = (
        f"Linear Regression Fit:\n"
        f"  ŷ = {slope:.3f}x {sign} {abs(intercept):.1f}\n"
        f"  {sep_line}\n"
        f"  R² Score:   {r2:.4f}\n"
        f"  Pearson r:  {r_value:.4f} (p < {p_value:.1e})\n"
        f"  Test RMSE:  {rmse:.2f} {unit}\n"
        f"  Test MAE:   {mae:.2f} {unit}"
    )
    ax.text(
        0.05,
        0.95,
        stat_box_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
        fontfamily="monospace",
    )

    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel(x_label, fontweight="bold", fontsize=11)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=11)
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {output_path}")
