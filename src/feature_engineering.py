"""Feature Engineering Module for Predictive Maintenance.

1. NASA C-MAPSS: Strictly backward-looking temporal feature engineering:
   - Rolling means and standard deviations across windows [5, 10, 20]
   - First-order backward difference (rate of change)
   - Baseline deviation from initial engine cycles (first 5 cycles baseline)
   - Zero future information leakage.

2. UCI AI4I 2020: Domain-informed physical feature engineering based on machine physics:
   - Mechanical power: P = Torque * (2 * pi * RPM / 60) [W]
   - Temperature difference: delta_T = Process Temp - Air Temp [K]
   - Temperature ratio: Process Temp / Air Temp
   - Torque-to-Speed ratio: Torque / RPM
   - Overstrain indicator: Torque * Tool wear
   - Heat dissipation index: delta_T * RPM
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def engineer_nasa_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    rolling_windows: List[int] = [5, 10, 20],
    baseline_cycles: int = 5
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate backward-looking temporal features for NASA engine trajectories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engine trajectories with 'unit_id', 'cycle', and sensor measurements.
    sensor_cols : List[str], optional
        Sensor columns to compute temporal features on.
    rolling_windows : List[int]
        Window lengths for rolling statistics (must be backward-looking).
    baseline_cycles : int
        Number of initial cycles to define baseline healthy state per engine.
        
    Returns
    -------
    pd.DataFrame, List[str]
        DataFrame with engineered features and list of engineered feature names.
    """
    df_out = df.copy()

    # Determine sensor columns if not provided
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]

    engineered_feature_names = []

    new_cols_dict = {}

    # 1. Baseline Deviation
    # Compute mean for first `baseline_cycles` per engine
    baseline_df = (
        df_out[df_out["cycle"] <= baseline_cycles]
        .groupby("unit_id")[sensor_cols]
        .mean()
        .reset_index()
    )
    
    baseline_merged = df_out[["unit_id"]].merge(baseline_df, on="unit_id", how="left")
    baseline_merged.index = df_out.index
    
    for col in sensor_cols:
        col_name = f"{col}_baseline_dev"
        new_cols_dict[col_name] = (df_out[col] - baseline_merged[col]).fillna(0.0)
        engineered_feature_names.append(col_name)

    # 2. Group by unit_id to strictly prevent cross-engine feature contamination
    grouped = df_out.groupby("unit_id")

    # 3. First-order backward differences (Lag 1)
    for col in sensor_cols:
        diff_col = f"{col}_diff1"
        new_cols_dict[diff_col] = grouped[col].diff(1).fillna(0.0)
        engineered_feature_names.append(diff_col)

    # 4. Backward-looking rolling mean and rolling std
    for window in rolling_windows:
        for col in sensor_cols:
            mean_col = f"{col}_roll_mean_{window}"
            std_col = f"{col}_roll_std_{window}"
            
            # min_periods=1 ensures no NaNs at the beginning of the trajectory
            new_cols_dict[mean_col] = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            new_cols_dict[std_col] = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0.0)
                .reset_index(level=0, drop=True)
            )
            engineered_feature_names.extend([mean_col, std_col])

    # Concatenate all new columns at once to prevent fragmentation
    new_features_df = pd.DataFrame(new_cols_dict, index=df_out.index)
    df_out = pd.concat([df_out, new_features_df], axis=1)

    return df_out, engineered_feature_names


def engineer_ai4i_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute domain-informed physical features for the UCI AI4I 2020 dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input AI4I raw DataFrame.
        
    Returns
    -------
    pd.DataFrame, List[str]
        DataFrame with physical features added and list of new feature names.
    """
    df_out = df.copy()
    new_features = []

    # 1. Angular velocity in rad/s: omega = 2 * pi * RPM / 60
    omega = 2.0 * np.pi * df_out["Rotational speed [rpm]"] / 60.0
    
    # Mechanical Power [W]: P = Torque [Nm] * omega [rad/s]
    df_out["Mechanical power [W]"] = df_out["Torque [Nm]"] * omega
    new_features.append("Mechanical power [W]")

    # 2. Temperature difference [K]: Process Temp - Air Temp
    df_out["Temp difference [K]"] = (
        df_out["Process temperature [K]"] - df_out["Air temperature [K]"]
    )
    new_features.append("Temp difference [K]")

    # 3. Temperature ratio: Process Temp / Air Temp
    df_out["Temp ratio"] = (
        df_out["Process temperature [K]"] / df_out["Air temperature [K]"]
    )
    new_features.append("Temp ratio")

    # 4. Torque to Rotational Speed ratio
    df_out["Torque to speed ratio"] = (
        df_out["Torque [Nm]"] / (df_out["Rotational speed [rpm]"] + 1e-5)
    )
    new_features.append("Torque to speed ratio")

    # 5. Overstrain Indicator: Torque * Tool wear (Physical cause of Overstrain Failure OSF)
    df_out["Overstrain indicator [Nm*min]"] = (
        df_out["Torque [Nm]"] * df_out["Tool wear [min]"]
    )
    new_features.append("Overstrain indicator [Nm*min]")

    # 6. Heat Dissipation Indicator: Temp difference * Rotational speed
    # (Heat Dissipation Failure HDF occurs when delta_T < 8.6 K and RPM < 1380)
    df_out["Heat dissipation index"] = (
        df_out["Temp difference [K]"] * df_out["Rotational speed [rpm]"]
    )
    new_features.append("Heat dissipation index")

    # 7. Power Wear Product: Mechanical Power * Tool wear
    df_out["Power wear index"] = (
        df_out["Mechanical power [W]"] * df_out["Tool wear [min]"]
    )
    new_features.append("Power wear index")

    return df_out, new_features


if __name__ == "__main__":
    from nasa_loader import load_nasa_subset
    from data_cleaning import clean_nasa_data
    from rul_generation import compute_training_rul
    from ai4i_loader import load_ai4i_raw

    train_df, _, _ = load_nasa_subset("FD001", raw_dir="data/raw/nasa_cmapss")
    train_df = compute_training_rul(train_df)
    clean_train, _ = clean_nasa_data(train_df)
    
    sens_cols = [c for c in clean_train.columns if c.startswith("sensor_") or c.startswith("setting_")]
    engineered_nasa, new_cols_nasa = engineer_nasa_features(clean_train, sensor_cols=sens_cols)
    print(f"NASA Original clean features: {len(sens_cols)}")
    print(f"NASA Engineered features added: {len(new_cols_nasa)}")
    print(f"NASA Total columns: {engineered_nasa.shape[1]}")
    print(f"Any NaNs in NASA features: {engineered_nasa[new_cols_nasa].isna().any().any()}")

    ai4i_df = load_ai4i_raw("data/raw/ai4i/ai4i2020.csv")
    engineered_ai4i, new_cols_ai4i = engineer_ai4i_features(ai4i_df)
    print(f"\nAI4I Engineered features added: {len(new_cols_ai4i)}")
    print(f"AI4I Engineered features list: {new_cols_ai4i}")
    print(f"Any NaNs in AI4I features: {engineered_ai4i[new_cols_ai4i].isna().any().any()}")
