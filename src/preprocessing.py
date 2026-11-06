"""Complete End-to-End Preprocessing Pipeline Module.

Orchestrates:
1. NASA C-MAPSS Preprocessing:
   - Trajectory RUL labeling and early-warning generation
   - Zero-variance sensor identification and pruning
   - Engine-level train/validation splitting (strictly by engine ID, seed=42)
   - Backward-looking temporal feature engineering
   - MinMax / Standard scaling fitted ONLY on training engines
   - Transformation of validation and official test engines
   - 3D temporal sequence construction across windows [30, 50, 60] saved to NPZ
   - Saving fitted scalers via joblib

2. UCI AI4I 2020 Preprocessing:
   - Domain-informed physical feature engineering
   - Target and leakage column isolation
   - Stratified 80:20 train-test split (seed=42)
   - One-hot encoding of categorical features and standardization of numerical features
   - Fitted ColumnTransformer / Pipeline saved via joblib
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import yaml

from nasa_loader import load_nasa_subset
from rul_generation import compute_training_rul, reconstruct_test_rul, add_early_warning_targets
from data_cleaning import clean_nasa_data, clean_ai4i_data
from feature_engineering import engineer_nasa_features, engineer_ai4i_features
from sequence_generation import create_engine_sequences, save_sequences_npz


def split_nasa_engines(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    random_seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split engine IDs into training and validation groups without engine mixing."""
    unique_engines = np.sort(df["unit_id"].unique())
    rng = np.random.RandomState(random_seed)
    shuffled = rng.permutation(unique_engines)
    
    n_train = int(len(shuffled) * train_ratio)
    train_engine_ids = sorted(shuffled[:n_train].tolist())
    val_engine_ids = sorted(shuffled[n_train:].tolist())

    # Guarantee zero overlap
    overlap = set(train_engine_ids).intersection(set(val_engine_ids))
    if overlap:
        raise ValueError(f"Engine split leakage detected! Overlapping engines: {overlap}")

    return train_engine_ids, val_engine_ids


def process_nasa_subset(
    subset: str = "FD001",
    raw_dir: Union[str, Path] = "data/raw/nasa_cmapss",
    processed_dir: Union[str, Path] = "data/processed/nasa",
    models_dir: Union[str, Path] = "models/preprocessing",
    window_lengths: List[int] = [30, 50, 60],
    train_val_ratio: float = 0.80,
    random_seed: int = 42,
    scaling_method: str = "minmax"
) -> Dict[str, Any]:
    """Execute complete preprocessing workflow for a NASA C-MAPSS subset."""
    print(f"\n{'='*60}\nPROCESSING NASA SUBSET: {subset}\n{'='*60}")
    raw_path = Path(raw_dir)
    proc_subset_dir = Path(processed_dir) / subset
    models_path = Path(models_dir)
    proc_subset_dir.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data and test RUL
    train_raw, test_raw, test_rul_vec = load_nasa_subset(subset, raw_dir=raw_path)

    # 2. Compute RUL and early warning targets
    train_labeled = compute_training_rul(train_raw)
    train_labeled = add_early_warning_targets(train_labeled, [10, 20, 30])

    test_labeled = reconstruct_test_rul(test_raw, test_rul_vec)
    test_labeled = add_early_warning_targets(test_labeled, [10, 20, 30])

    # 3. Clean: detect zero-variance sensors on train
    clean_train_full, clean_audit = clean_nasa_data(train_labeled, dataset_name=f"{subset} Train")
    dropped_cols = clean_audit["dropped_columns"]

    # Drop same columns from test
    clean_test, _ = clean_nasa_data(test_labeled, columns_to_drop=dropped_cols, dataset_name=f"{subset} Test")

    # 4. Engine-level split on training trajectories
    train_engines, val_engines = split_nasa_engines(
        clean_train_full, train_ratio=train_val_ratio, random_seed=random_seed
    )
    print(f"Engine split: {len(train_engines)} train engines, {len(val_engines)} val engines.")

    train_df = clean_train_full[clean_train_full["unit_id"].isin(train_engines)].copy()
    val_df = clean_train_full[clean_train_full["unit_id"].isin(val_engines)].copy()

    # 5. Temporal Feature Engineering (backward-looking strictly per engine)
    base_sensor_cols = [
        c for c in train_df.columns
        if c not in ["unit_id", "cycle", "RUL", "early_warning_10", "early_warning_20", "early_warning_30"]
    ]

    print("Engineering temporal features on train engines...")
    train_eng, eng_feature_names = engineer_nasa_features(train_df, sensor_cols=base_sensor_cols)
    print("Engineering temporal features on validation engines...")
    val_eng, _ = engineer_nasa_features(val_df, sensor_cols=base_sensor_cols)
    print("Engineering temporal features on test engines...")
    test_eng, _ = engineer_nasa_features(clean_test, sensor_cols=base_sensor_cols)

    all_features = base_sensor_cols + eng_feature_names

    # 6. Scaling: Fit ONLY on training engines!
    print(f"Fitting {scaling_method} scaler on training engines only...")
    if scaling_method == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = StandardScaler()

    scaler.fit(train_eng[all_features])

    # Transform copies
    train_scaled = train_eng.copy()
    val_scaled = val_eng.copy()
    test_scaled = test_eng.copy()

    train_scaled[all_features] = scaler.transform(train_eng[all_features])
    val_scaled[all_features] = scaler.transform(val_eng[all_features])
    test_scaled[all_features] = scaler.transform(test_eng[all_features])

    # Save fitted scaler
    scaler_path = models_path / f"nasa_scaler_{subset}.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")

    # Save scaled tabular datasets
    train_scaled.to_csv(proc_subset_dir / "train_tabular.csv", index=False)
    val_scaled.to_csv(proc_subset_dir / "val_tabular.csv", index=False)
    test_scaled.to_csv(proc_subset_dir / "test_tabular.csv", index=False)
    print(f"Saved scaled tabular datasets to: {proc_subset_dir}")

    # Save engine split metadata
    split_meta = {
        "subset": subset,
        "total_train_engines": int(clean_train_full["unit_id"].nunique()),
        "train_engines": train_engines,
        "val_engines": val_engines,
        "test_engines": sorted(clean_test["unit_id"].unique().tolist()),
        "base_features": base_sensor_cols,
        "engineered_features": eng_feature_names,
        "all_features": all_features,
        "dropped_sensors": dropped_cols,
    }
    with open(proc_subset_dir / "metadata.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # 7. Generate Sequences across window lengths
    seq_summary = {}
    for w in window_lengths:
        print(f"Constructing 3D sliding sequences for window_length={w}...")
        train_seq = create_engine_sequences(train_scaled, feature_cols=all_features, window_length=w)
        val_seq = create_engine_sequences(val_scaled, feature_cols=all_features, window_length=w)
        test_seq = create_engine_sequences(test_scaled, feature_cols=all_features, window_length=w)

        save_sequences_npz(train_seq, all_features, w, proc_subset_dir / f"sequences_w{w}_train.npz")
        save_sequences_npz(val_seq, all_features, w, proc_subset_dir / f"sequences_w{w}_val.npz")
        save_sequences_npz(test_seq, all_features, w, proc_subset_dir / f"sequences_w{w}_test.npz")

        seq_summary[f"w{w}"] = {
            "train_sequences_shape": list(train_seq["X"].shape),
            "val_sequences_shape": list(val_seq["X"].shape),
            "test_sequences_shape": list(test_seq["X"].shape),
        }
        print(f"  w={w}: Train shape={train_seq['X'].shape}, Val shape={val_seq['X'].shape}, Test shape={test_seq['X'].shape}")

    print(f"Completed preprocessing for {subset}.")
    return {
        "subset": subset,
        "split_meta": split_meta,
        "seq_summary": seq_summary,
        "scaler_path": str(scaler_path),
    }


def process_ai4i_dataset(
    raw_path: Union[str, Path] = "data/raw/ai4i/ai4i2020.csv",
    processed_dir: Union[str, Path] = "data/processed/ai4i",
    models_dir: Union[str, Path] = "models/preprocessing",
    test_size: float = 0.20,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Execute complete preprocessing workflow for UCI AI4I 2020 dataset."""
    print(f"\n{'='*60}\nPROCESSING UCI AI4I 2020 DATASET\n{'='*60}")
    proc_dir = Path(processed_dir)
    models_path = Path(models_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    from ai4i_loader import load_ai4i_raw

    # 1. Load raw
    df_raw = load_ai4i_raw(raw_path)

    # 2. Clean
    df_clean, clean_audit = clean_ai4i_data(df_raw)
    df_clean.to_csv(proc_dir / "ai4i_cleaned.csv", index=False)

    # 3. Domain-informed physical feature engineering
    df_eng, new_features = engineer_ai4i_features(df_clean)
    df_eng.to_csv(proc_dir / "ai4i_engineered.csv", index=False)
    print(f"Saved engineered dataset with {len(new_features)} domain physical features.")

    # 4. Target and leakage column isolation
    # Strictly exclude failure modes and identifiers from predictive input features
    target_col = "Machine failure"
    id_cols = ["UDI", "Product ID"]
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    exclude_cols = id_cols + [target_col] + failure_modes

    feature_cols = [c for c in df_eng.columns if c not in exclude_cols]
    cat_cols = ["Type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df_eng[feature_cols]
    y = df_eng[target_col]

    # 5. Stratified 80:20 Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    print(f"Stratified Split: Train records={len(X_train_raw)}, Test records={len(X_test_raw)}")
    print(f"Train failure rate: {y_train.mean():.4f}, Test failure rate: {y_test.mean():.4f}")

    # 6. Fit Preprocessor ONLY on training set!
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    preprocessor.fit(X_train_raw)

    # Extract transformed column names
    ohe_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    transformed_feature_names = num_cols + ohe_cols

    X_train_proc = pd.DataFrame(
        preprocessor.transform(X_train_raw),
        columns=transformed_feature_names,
        index=X_train_raw.index
    )
    X_test_proc = pd.DataFrame(
        preprocessor.transform(X_test_raw),
        columns=transformed_feature_names,
        index=X_test_raw.index
    )

    # Save fitted pipeline
    pipeline_path = models_path / "ai4i_preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)
    print(f"Saved AI4I preprocessor to: {pipeline_path}")

    # Save X and y splits
    X_train_proc.to_csv(proc_dir / "X_train.csv", index=False)
    X_test_proc.to_csv(proc_dir / "X_test.csv", index=False)
    y_train.to_frame(name=target_col).to_csv(proc_dir / "y_train.csv", index=False)
    y_test.to_frame(name=target_col).to_csv(proc_dir / "y_test.csv", index=False)

    # Save metadata
    meta = {
        "target": target_col,
        "input_features": transformed_feature_names,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "excluded_leakage_features": exclude_cols,
        "train_samples": len(X_train_proc),
        "test_samples": len(X_test_proc),
        "train_failure_count": int(y_train.sum()),
        "test_failure_count": int(y_test.sum()),
    }
    with open(proc_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Completed AI4I preprocessing.")
    return meta


def main():
    """Execute complete Phase 1 preprocessing."""
    config_file = Path("config/config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 1. Process NASA FD001 (initial benchmark)
    nasa_result_fd001 = process_nasa_subset(
        subset="FD001",
        raw_dir="data/raw/nasa_cmapss",
        processed_dir="data/processed/nasa",
        models_dir="models/preprocessing",
        window_lengths=[30, 50, 60],
        train_val_ratio=0.80,
        random_seed=42,
        scaling_method="minmax"
    )

    # Also build pipeline support for FD002, FD003, FD004 as required by Section 1:
    # "Use FD001 as the initial development benchmark. Build the pipeline so it can subsequently support FD002, FD003 and FD004."
    for sub in ["FD002", "FD003", "FD004"]:
        process_nasa_subset(
            subset=sub,
            raw_dir="data/raw/nasa_cmapss",
            processed_dir="data/processed/nasa",
            models_dir="models/preprocessing",
            window_lengths=[30],  # create benchmark sequences for all subsets
            train_val_ratio=0.80,
            random_seed=42,
            scaling_method="minmax"
        )

    # 2. Process AI4I 2020
    ai4i_result = process_ai4i_dataset(
        raw_path="data/raw/ai4i/ai4i2020.csv",
        processed_dir="data/processed/ai4i",
        models_dir="models/preprocessing",
        test_size=0.20,
        random_seed=42
    )

    print("\n" + "="*60)
    print("ALL PREPROCESSING PIPELINES EXECUTED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
