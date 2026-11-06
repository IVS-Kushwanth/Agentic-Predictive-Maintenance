"""Temporal Sequence Construction Module for Deep Learning Architectures.

Constructs 3D multivariate temporal sliding-window sequences from NASA engine trajectories.
Ensures zero window overlap across engine boundaries and zero target leakage.

Output Shape: (num_sequences, window_length, num_features)
Storage Format: Compressed NumPy archive (.npz)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def create_engine_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_length: int = 30,
    stride: int = 1,
    min_length_policy: str = "pad"  # 'pad' or 'skip'
) -> Dict[str, np.ndarray]:
    """Construct sliding-window temporal sequences strictly within individual engine trajectories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engine trajectories. Must include 'unit_id', 'cycle', 'RUL',
        and the specified feature_cols.
    feature_cols : List[str]
        List of numeric feature column names to include in each time step.
    window_length : int
        Number of consecutive operational cycles per sequence (L).
    stride : int
        Step size between consecutive sliding windows (default 1).
    min_length_policy : str
        Policy for trajectories shorter than window_length ('pad' with earliest observation or 'skip').
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'X': 3D array of shape (N, window_length, len(feature_cols))
        - 'y_rul': 1D array of terminal RUL targets of shape (N,)
        - 'y_ew10': 1D array of terminal early-warning (10 cycles) of shape (N,)
        - 'y_ew20': 1D array of terminal early-warning (20 cycles) of shape (N,)
        - 'y_ew30': 1D array of terminal early-warning (30 cycles) of shape (N,)
        - 'engine_ids': 1D array of engine unit_id for each sequence of shape (N,)
        - 'end_cycles': 1D array of cycle at the end of the sequence of shape (N,)
    """
    sequences_X = []
    targets_rul = []
    targets_ew10 = []
    targets_ew20 = []
    targets_ew30 = []
    seq_engine_ids = []
    seq_end_cycles = []

    # Sort to guarantee temporal order
    df_sorted = df.sort_values(by=["unit_id", "cycle"])

    for unit_id, engine_df in df_sorted.groupby("unit_id"):
        features = engine_df[feature_cols].to_numpy(dtype=np.float32)
        ruls = engine_df["RUL"].to_numpy(dtype=np.float32)
        cycles = engine_df["cycle"].to_numpy(dtype=np.int32)
        
        # Check early warning targets if present
        ew10 = engine_df["early_warning_10"].to_numpy(dtype=np.int32) if "early_warning_10" in engine_df else (ruls <= 10).astype(np.int32)
        ew20 = engine_df["early_warning_20"].to_numpy(dtype=np.int32) if "early_warning_20" in engine_df else (ruls <= 20).astype(np.int32)
        ew30 = engine_df["early_warning_30"].to_numpy(dtype=np.int32) if "early_warning_30" in engine_df else (ruls <= 30).astype(np.int32)

        n_cycles = len(engine_df)

        if n_cycles < window_length:
            if min_length_policy == "pad":
                # Left-pad with first observation to maintain length
                pad_width = window_length - n_cycles
                padded_features = np.pad(features, ((pad_width, 0), (0, 0)), mode="edge")
                sequences_X.append(padded_features)
                targets_rul.append(ruls[-1])
                targets_ew10.append(ew10[-1])
                targets_ew20.append(ew20[-1])
                targets_ew30.append(ew30[-1])
                seq_engine_ids.append(unit_id)
                seq_end_cycles.append(cycles[-1])
            continue

        # Extract sliding windows
        for start_idx in range(0, n_cycles - window_length + 1, stride):
            end_idx = start_idx + window_length
            window_slice = features[start_idx:end_idx]
            
            # Target is the value at the final cycle of the window
            terminal_idx = end_idx - 1
            
            sequences_X.append(window_slice)
            targets_rul.append(ruls[terminal_idx])
            targets_ew10.append(ew10[terminal_idx])
            targets_ew20.append(ew20[terminal_idx])
            targets_ew30.append(ew30[terminal_idx])
            seq_engine_ids.append(unit_id)
            seq_end_cycles.append(cycles[terminal_idx])

    if len(sequences_X) == 0:
        return {
            "X": np.empty((0, window_length, len(feature_cols)), dtype=np.float32),
            "y_rul": np.empty((0,), dtype=np.float32),
            "y_ew10": np.empty((0,), dtype=np.int32),
            "y_ew20": np.empty((0,), dtype=np.int32),
            "y_ew30": np.empty((0,), dtype=np.int32),
            "engine_ids": np.empty((0,), dtype=np.int32),
            "end_cycles": np.empty((0,), dtype=np.int32),
        }

    return {
        "X": np.asarray(sequences_X, dtype=np.float32),
        "y_rul": np.asarray(targets_rul, dtype=np.float32),
        "y_ew10": np.asarray(targets_ew10, dtype=np.int32),
        "y_ew20": np.asarray(targets_ew20, dtype=np.int32),
        "y_ew30": np.asarray(targets_ew30, dtype=np.int32),
        "engine_ids": np.asarray(seq_engine_ids, dtype=np.int32),
        "end_cycles": np.asarray(seq_end_cycles, dtype=np.int32),
    }


def save_sequences_npz(
    sequence_data: Dict[str, np.ndarray],
    feature_names: List[str],
    window_length: int,
    output_path: Union[str, Path]
) -> Path:
    """Save sequence arrays and metadata to an optimized .npz archive."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        X=sequence_data["X"],
        y_rul=sequence_data["y_rul"],
        y_ew10=sequence_data["y_ew10"],
        y_ew20=sequence_data["y_ew20"],
        y_ew30=sequence_data["y_ew30"],
        engine_ids=sequence_data["engine_ids"],
        end_cycles=sequence_data["end_cycles"],
        feature_names=np.array(feature_names, dtype=str),
        window_length=np.array([window_length], dtype=np.int32),
    )
    return out_file


def load_sequences_npz(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and verify a stored sequence NPZ archive."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    return {
        "X": data["X"],
        "y_rul": data["y_rul"],
        "y_ew10": data["y_ew10"],
        "y_ew20": data["y_ew20"],
        "y_ew30": data["y_ew30"],
        "engine_ids": data["engine_ids"],
        "end_cycles": data["end_cycles"],
        "feature_names": data["feature_names"].tolist(),
        "window_length": int(data["window_length"][0]),
    }


if __name__ == "__main__":
    from typing import Any
    from nasa_loader import load_nasa_subset
    from data_cleaning import clean_nasa_data
    from rul_generation import compute_training_rul, add_early_warning_targets

    train_df, _, _ = load_nasa_subset("FD001", raw_dir="data/raw/nasa_cmapss")
    train_df = compute_training_rul(train_df)
    train_df = add_early_warning_targets(train_df, [10, 20, 30])
    clean_train, _ = clean_nasa_data(train_df)
    
    feature_cols = [c for c in clean_train.columns if c not in ["unit_id", "cycle", "RUL", "early_warning_10", "early_warning_20", "early_warning_30"]]
    
    seqs = create_engine_sequences(clean_train, feature_cols=feature_cols, window_length=30)
    print(f"Window length: 30")
    print(f"X shape: {seqs['X'].shape}")
    print(f"y_rul shape: {seqs['y_rul'].shape}")
    print(f"y_ew30 mean (positive warning rate): {seqs['y_ew30'].mean():.4f}")
    print(f"Engine IDs represented: {len(np.unique(seqs['engine_ids']))}")
