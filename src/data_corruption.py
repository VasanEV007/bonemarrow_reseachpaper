"""
data_corruption.py - Artificial Data Corruption ("Dirty Data" Step)
Injects artificial noise, missingness, duplicates, messy string casing, and extreme outliers.
"""

import numpy as np
import pandas as pd


def inject_noise_and_corruption(
    df: pd.DataFrame,
    missing_fraction: float = 0.12,
    num_duplicates: int = 12,
    num_outliers: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Creates a 'dirty' corrupted version of the training dataset:
    1. Randomly replaces 10-15% of numerical values with NaN / '?'
    2. Introduces duplicate rows
    3. Injects inconsistent string formatting/casing into categorical columns
    4. Inserts extreme numerical outliers
    """
    np.random.seed(random_state)
    dirty_df = df.copy()

    # Exclude id and target from corruption to preserve primary identity & truth
    protected_cols = ["id", "survival_status"]
    candidate_cols = [c for c in dirty_df.columns if c not in protected_cols]

    # Identify continuous numerical columns
    numeric_cols = []
    for col in candidate_cols:
        # Check if column is numeric or can be numeric
        converted = pd.to_numeric(dirty_df[col].replace("?", np.nan), errors="coerce")
        if converted.notnull().sum() > 0.5 * len(dirty_df) and converted.nunique() > 5:
            numeric_cols.append(col)

    # 1. Inject missingness (NaN) in 10-15% of numerical values
    for col in numeric_cols:
        mask = np.random.rand(len(dirty_df)) < missing_fraction
        dirty_df.loc[mask, col] = np.nan

    # 2. Inject inconsistent string casing / whitespace in categorical fields
    if "Disease" in dirty_df.columns:
        casing_variants = {
            "AML": ["aml", "  AML  ", "AmL", "aml "],
            "ALL": ["  all ", "All", "ALL", "aLl  "],
            "chronic": ["CHRONIC", " Chronic ", "chrOnic", "CHRONIC "],
            "nonmalignant": ["Non-Malignant", "NONMALIGNANT", " nonmalignant ", "non-malignant"],
            "lymphoma": ["Lymphoma", "LYMPHOMA", " lymphoma ", "LyMpHoma"],
        }
        for idx in dirty_df.index:
            val = str(dirty_df.at[idx, "Disease"]).strip()
            if val in casing_variants and np.random.rand() < 0.45:
                variants = casing_variants[val]
                dirty_df.at[idx, "Disease"] = np.random.choice(variants)

    # 3. Insert extreme numerical outliers
    outlier_targets = [c for c in ["CD34kgx10d6", "CD3dkgx10d8", "Rbodymass", "Donorage"] if c in numeric_cols]
    for _ in range(num_outliers):
        target_col = np.random.choice(outlier_targets)
        target_idx = np.random.choice(dirty_df.index)
        # Extreme factor (very high or unphysical values)
        multiplier = np.random.choice([15.0, 25.0, -10.0, 100.0])
        curr_val = pd.to_numeric(dirty_df.at[target_idx, target_col], errors="coerce")
        if pd.notnull(curr_val) and curr_val != 0:
            dirty_df.at[target_idx, target_col] = round(curr_val * multiplier, 2)
        else:
            dirty_df.at[target_idx, target_col] = 9999.0

    # 4. Introduce duplicate rows
    dup_indices = np.random.choice(dirty_df.index, size=num_duplicates, replace=True)
    duplicates = dirty_df.loc[dup_indices].copy()
    # Keep identical feature values to simulate real duplicates
    dirty_df = pd.concat([dirty_df, duplicates], ignore_index=True)

    return dirty_df
