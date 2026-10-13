"""
evaluation.py - Evaluation Metrics, Confusion Matrices, and Benchmark Reporting
Computes Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa, RMSE, MAE, and generates Markdown tables.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def compute_comprehensive_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_or_cont: Optional[np.ndarray] = None,
    training_time_sec: float = 0.0,
) -> Dict:
    """
    Computes all standard classification and regression metrics:
    Accuracy, Precision, Recall, F1-Score, Matthews Correlation Coefficient (MCC),
    Cohen's Kappa, RMSE, MAE, and Training Time.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # For RMSE/MAE, use continuous values or probabilities if provided, else binary predictions
    score_vector = y_prob_or_cont if y_prob_or_cont is not None else y_pred
    mae = mean_absolute_error(y_true, score_vector)
    mse = mean_squared_error(y_true, score_vector)
    rmse = np.sqrt(mse)

    return {
        "Model_Name": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
        "Cohens_Kappa": round(float(kappa), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "Training_Time_Sec": round(float(training_time_sec), 3),
    }


def compute_confusion_matrix_breakdown(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict:
    """Computes TN, FP, FN, TP, Sensitivity, Specificity, PPV, NPV."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "Model": model_name,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity": round(float(sensitivity), 4),
        "Specificity": round(float(specificity), 4),
        "PPV": round(float(ppv), 4),
        "NPV": round(float(npv), 4),
    }


def format_markdown_table(df: pd.DataFrame, title: str) -> str:
    """Formats a pandas DataFrame into a clean Markdown table with title without requiring tabulate."""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        data_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    table_md = "\n".join([header_line, sep_line] + data_lines)
    return f"\n### {title}\n\n{table_md}\n"
