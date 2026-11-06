"""Data Validation and Quality Assurance Module.

Performs rigorous statistical and structural validation checks on:
1. NASA C-MAPSS engine degradation datasets (FD001-FD004)
2. UCI AI4I 2020 Predictive Maintenance dataset
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def validate_nasa_dataset(df: pd.DataFrame, dataset_name: str = "NASA C-MAPSS") -> Dict[str, Any]:
    """Execute comprehensive data quality checks on a NASA C-MAPSS DataFrame."""
    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "unique_engines": int(df["unit_id"].nunique()),
        "passed_all_critical_checks": True,
        "critical_issues": [],
        "warnings": [],
    }

    # 1. Missing Values Check
    missing_count = int(df.isna().sum().sum())
    report["missing_values_total"] = missing_count
    if missing_count > 0:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(f"Found {missing_count} missing (NaN) values.")

    # 2. Infinite / Invalid Numerical Values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    is_inf = np.isinf(df[numeric_cols].to_numpy()).sum()
    report["infinite_values_total"] = int(is_inf)
    if is_inf > 0:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(f"Found {is_inf} infinite numerical values.")

    # 3. Duplicate Records Check (unit_id, cycle)
    duplicates = int(df.duplicated(subset=["unit_id", "cycle"]).sum())
    report["duplicate_engine_cycle_pairs"] = duplicates
    if duplicates > 0:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(f"Found {duplicates} duplicate (unit_id, cycle) pairs.")

    # 4. Cycle Ordering & Trajectory Continuity
    continuity_issues = []
    min_cycles = []
    max_cycles = []
    trajectory_lengths = {}

    for unit_id, group in df.groupby("unit_id"):
        cycles = group["cycle"].to_numpy()
        min_cycles.append(int(cycles.min()))
        max_cycles.append(int(cycles.max()))
        trajectory_lengths[int(unit_id)] = len(cycles)

        # Check strictly monotonic increasing
        if not np.all(np.diff(cycles) == 1):
            continuity_issues.append(int(unit_id))

    report["cycle_continuity_violations"] = continuity_issues
    if continuity_issues:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(
            f"Non-contiguous cycle progression detected in engines: {continuity_issues[:5]}"
        )

    report["min_cycle_observed"] = int(min(min_cycles))
    report["max_cycle_observed"] = int(max(max_cycles))
    report["shortest_trajectory_length"] = int(min(trajectory_lengths.values()))
    report["longest_trajectory_length"] = int(max(trajectory_lengths.values()))
    report["mean_trajectory_length"] = float(np.round(np.mean(list(trajectory_lengths.values())), 2))
    report["median_trajectory_length"] = float(np.median(list(trajectory_lengths.values())))

    # 5. Constant and Near-Constant Sensors
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]
    constant_sensors = []
    near_constant_sensors = []  # std < 1e-4

    sensor_stats = {}
    for col in sensor_cols:
        col_std = float(df[col].std())
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
        col_nunique = int(df[col].nunique())

        sensor_stats[col] = {
            "mean": round(col_mean, 4),
            "std": round(col_std, 4),
            "min": round(col_min, 4),
            "max": round(col_max, 4),
            "unique_values": col_nunique,
        }

        if col_nunique <= 1 or col_std == 0:
            constant_sensors.append(col)
        elif col_std < 1e-4:
            near_constant_sensors.append(col)

    report["constant_sensors"] = constant_sensors
    report["near_constant_sensors"] = near_constant_sensors
    report["sensor_statistics"] = sensor_stats

    if constant_sensors:
        report["warnings"].append(f"Sensors with zero variance across all records: {constant_sensors}")

    return report


def validate_ai4i_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute quality checks on the UCI AI4I 2020 dataset."""
    report: Dict[str, Any] = {
        "dataset_name": "UCI AI4I 2020 Predictive Maintenance",
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "passed_all_critical_checks": True,
        "critical_issues": [],
        "warnings": [],
    }

    # Missing values
    missing = int(df.isna().sum().sum())
    report["missing_values_total"] = missing
    if missing > 0:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(f"Found {missing} missing values.")

    # Duplicate rows
    duplicates = int(df.duplicated(subset=["UDI"]).sum())
    report["duplicate_udi_count"] = duplicates
    if duplicates > 0:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].append(f"Found {duplicates} duplicate UDI entries.")

    # Range and domain physical bounds checks
    range_issues = []
    if (df["Air temperature [K]"] <= 0).any():
        range_issues.append("Air temperature contains non-positive Kelvin values.")
    if (df["Process temperature [K]"] <= 0).any():
        range_issues.append("Process temperature contains non-positive Kelvin values.")
    if (df["Rotational speed [rpm]"] <= 0).any():
        range_issues.append("Rotational speed contains non-positive RPM values.")
    if (df["Torque [Nm]"] < 0).any():
        range_issues.append("Torque contains negative Nm values.")
    if (df["Tool wear [min]"] < 0).any():
        range_issues.append("Tool wear contains negative minutes.")

    report["physical_range_issues"] = range_issues
    if range_issues:
        report["passed_all_critical_checks"] = False
        report["critical_issues"].extend(range_issues)

    # Class balance and failure breakdown
    fail_counts = df["Machine failure"].value_counts().to_dict()
    fail_rate = float(df["Machine failure"].mean())
    report["class_distribution"] = {
        "negative_class_0": int(fail_counts.get(0, 0)),
        "positive_class_1": int(fail_counts.get(1, 0)),
        "failure_rate_percent": round(fail_rate * 100, 2),
    }

    # Failure modes
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_counts = {mode: int(df[mode].sum()) for mode in failure_modes}
    report["failure_mode_counts"] = mode_counts

    return report


if __name__ == "__main__":
    from nasa_loader import load_nasa_subset
    from ai4i_loader import load_ai4i_raw

    train_df, test_df, _ = load_nasa_subset("FD001", raw_dir="data/raw/nasa_cmapss")
    nasa_report = validate_nasa_dataset(train_df, "FD001 Train")
    print(f"NASA FD001 Train Passed Checks: {nasa_report['passed_all_critical_checks']}")
    print(f"NASA FD001 Constant Sensors: {nasa_report['constant_sensors']}")
    print(f"NASA FD001 Trajectory length: min={nasa_report['shortest_trajectory_length']}, max={nasa_report['longest_trajectory_length']}, mean={nasa_report['mean_trajectory_length']}")

    ai4i_df = load_ai4i_raw("data/raw/ai4i/ai4i2020.csv")
    ai4i_report = validate_ai4i_dataset(ai4i_df)
    print(f"\nAI4I Passed Checks: {ai4i_report['passed_all_critical_checks']}")
    print(f"AI4I Failure Rate: {ai4i_report['class_distribution']['failure_rate_percent']}%")
    print(f"AI4I Failure Modes: {ai4i_report['failure_mode_counts']}")
