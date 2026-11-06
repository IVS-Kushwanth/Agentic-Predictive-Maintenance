"""Remaining Useful Life (RUL) and Early Warning Target Generation Module.

Implements mathematically rigorous RUL calculation for both training and test trajectories
as well as configurable multi-horizon early-warning binary targets.

Formulas:
1. Training Engine Trajectory:
   RUL(engine, cycle) = max_cycle(engine) - cycle
   At final observed training cycle: RUL = 0.

2. Test Engine Trajectory Reconstruction:
   RUL(engine, cycle) = official_RUL(engine) + (max_test_cycle(engine) - cycle)
   At final observed test cycle: RUL = official_RUL(engine).

3. Early Warning Targets:
   early_warning_H = 1 if RUL <= H else 0, for H in [10, 20, 30]
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_training_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact ground truth RUL for each cycle in training trajectories."""
    df = train_df.copy()
    
    # Calculate maximum cycle per engine
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    
    # Ground truth RUL: max_cycle - current_cycle
    df["RUL"] = (max_cycles - df["cycle"]).astype(int)
    
    # Validation: at final cycle, RUL must be 0
    final_cycle_rul = df.groupby("unit_id").last()["RUL"]
    if not (final_cycle_rul == 0).all():
        raise ValueError("Assertion failed: Final training cycle RUL is not 0 for all engines.")

    return df


def reconstruct_test_rul(test_df: pd.DataFrame, test_rul_vector: np.ndarray) -> pd.DataFrame:
    """Reconstruct exact RUL trajectory for test engines using official terminal RUL values.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame containing test engine trajectories with columns ['unit_id', 'cycle', ...]
    test_rul_vector : np.ndarray
        Array containing the official remaining useful life at the final observed test cycle
        for each engine (index 0 corresponds to engine 1, etc.).
    
    Returns
    -------
    pd.DataFrame
        Test DataFrame with reconstructed RUL for every cycle.
    """
    df = test_df.copy()
    num_engines = df["unit_id"].nunique()
    
    if len(test_rul_vector) != num_engines:
        raise ValueError(
            f"Dimension mismatch: test_df has {num_engines} engines, "
            f"but test_rul_vector has {len(test_rul_vector)} labels."
        )

    # Engine IDs are 1-indexed in C-MAPSS
    unique_engines = np.sort(df["unit_id"].unique())
    engine_to_final_rul = {
        engine_id: test_rul_vector[idx]
        for idx, engine_id in enumerate(unique_engines)
    }

    # Find final observed cycle per test engine
    max_test_cycles = df.groupby("unit_id")["cycle"].transform("max")
    
    # Map the official final RUL for each row's engine
    official_rul_series = df["unit_id"].map(engine_to_final_rul)
    
    # Reconstruct RUL: official_final_RUL + (max_observed_cycle - current_cycle)
    df["RUL"] = (official_rul_series + (max_test_cycles - df["cycle"])).astype(int)

    # Validation: at final observed cycle, RUL must equal official_RUL
    final_rows = df.groupby("unit_id").last()
    for engine_id in unique_engines:
        actual_terminal_rul = final_rows.loc[engine_id, "RUL"]
        expected_terminal_rul = engine_to_final_rul[engine_id]
        if actual_terminal_rul != expected_terminal_rul:
            raise ValueError(
                f"Assertion failed for test engine {engine_id}: "
                f"calculated {actual_terminal_rul} != expected {expected_terminal_rul}"
            )

    return df


def add_early_warning_targets(
    df: pd.DataFrame,
    horizons: List[int] = [10, 20, 30]
) -> pd.DataFrame:
    """Add binary early-warning indicator targets based on RUL horizons.
    
    early_warning_H = 1 if RUL <= H else 0
    """
    if "RUL" not in df.columns:
        raise ValueError("DataFrame must contain 'RUL' column to compute early-warning targets.")

    df_out = df.copy()
    for h in horizons:
        col_name = f"early_warning_{h}"
        df_out[col_name] = (df_out["RUL"] <= h).astype(int)

    return df_out


def compute_rul_statistics(df: pd.DataFrame, label_type: str = "Training") -> Dict[str, Union[float, int]]:
    """Compute summary statistics for RUL distribution."""
    rul = df["RUL"]
    stats = {
        "dataset_type": label_type,
        "count_records": int(len(rul)),
        "min_rul": int(rul.min()),
        "max_rul": int(rul.max()),
        "mean_rul": float(np.round(rul.mean(), 2)),
        "std_rul": float(np.round(rul.std(), 2)),
        "median_rul": float(rul.median()),
        "q25_rul": float(rul.quantile(0.25)),
        "q75_rul": float(rul.quantile(0.75)),
    }
    return stats


if __name__ == "__main__":
    from nasa_loader import load_nasa_subset
    train_df, test_df, test_rul = load_nasa_subset("FD001", raw_dir="data/raw/nasa_cmapss")
    
    train_labeled = compute_training_rul(train_df)
    train_labeled = add_early_warning_targets(train_labeled, [10, 20, 30])
    print("Train labeled head:")
    print(train_labeled[["unit_id", "cycle", "RUL", "early_warning_10", "early_warning_20", "early_warning_30"]].head())

    test_labeled = reconstruct_test_rul(test_df, test_rul)
    test_labeled = add_early_warning_targets(test_labeled, [10, 20, 30])
    print("\nTest labeled head:")
    print(test_labeled[["unit_id", "cycle", "RUL", "early_warning_10", "early_warning_20", "early_warning_30"]].head())
    
    print("\nTrain RUL stats:", compute_rul_statistics(train_labeled, "Train FD001"))
    print("Test RUL stats:", compute_rul_statistics(test_labeled, "Test FD001"))
