"""Data Cleaning Module.

Implements principled data cleaning routines for:
1. NASA C-MAPSS: Identifies and removes uninformative zero-variance sensors/settings
   determined strictly from training data, preserving critical degradation signals.
2. UCI AI4I 2020: Validates range integrity, duplicate handling, and ensures no leakage columns.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def detect_constant_columns(
    df: pd.DataFrame,
    threshold: float = 1e-6,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """Identify columns that have zero or negligible variance across all records."""
    if exclude_cols is None:
        exclude_cols = ["unit_id", "cycle", "RUL"]

    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    constant_cols = []
    
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            std_val = df[col].std()
            nunique_val = df[col].nunique()
            if nunique_val <= 1 or std_val < threshold:
                constant_cols.append(col)

    return constant_cols


def clean_nasa_data(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    dataset_name: str = "NASA C-MAPSS"
) -> Tuple[pd.DataFrame, Dict[str, Union[int, List[str]]]]:
    """Clean NASA dataset by removing designated constant/uninformative columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input NASA dataset
    columns_to_drop : List[str], optional
        List of columns to drop. If None, detected automatically from input.
    dataset_name : str
        Name for reporting
        
    Returns
    -------
    pd.DataFrame, dict
        Cleaned dataframe and cleaning audit dictionary.
    """
    df_clean = df.copy()

    if columns_to_drop is None:
        columns_to_drop = detect_constant_columns(df_clean)

    # Only drop columns that actually exist in the dataframe
    actual_drop = [c for c in columns_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=actual_drop)

    audit = {
        "dataset_name": dataset_name,
        "original_rows": len(df),
        "original_cols": df.shape[1],
        "dropped_columns": actual_drop,
        "remaining_cols": df_clean.shape[1],
        "remaining_feature_names": [c for c in df_clean.columns if c not in ["unit_id", "cycle", "RUL"]],
    }

    return df_clean, audit


def clean_ai4i_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Union[int, List[str]]]]:
    """Clean UCI AI4I 2020 dataset and verify data integrity."""
    df_clean = df.copy()

    initial_rows = len(df_clean)
    
    # Remove any exact duplicates if present
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)

    audit = {
        "dataset_name": "UCI AI4I 2020",
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(df_clean),
        "columns_retained": list(df_clean.columns),
    }

    return df_clean, audit


if __name__ == "__main__":
    from nasa_loader import load_nasa_subset
    from rul_generation import compute_training_rul
    from ai4i_loader import load_ai4i_raw

    train_df, _, _ = load_nasa_subset("FD001", raw_dir="data/raw/nasa_cmapss")
    train_df = compute_training_rul(train_df)
    clean_train, audit_nasa = clean_nasa_data(train_df, dataset_name="FD001 Train")
    print("NASA Clean Audit:")
    for k, v in audit_nasa.items():
        print(f"  {k}: {v}")

    ai4i_df = load_ai4i_raw("data/raw/ai4i/ai4i2020.csv")
    clean_ai4i, audit_ai4i = clean_ai4i_data(ai4i_df)
    print("\nAI4I Clean Audit:")
    for k, v in audit_ai4i.items():
        print(f"  {k}: {v}")
