"""PyTorch Dataset and DataLoader Utilities for Predictive Maintenance.

Supports:
1. NASA C-MAPSS 3D Multivariate Sliding-Window Sequences.
   - Preserves engine trajectory boundaries.
   - Extracts terminal-cycle test sequences for PHM 2008 benchmark.
   - Provides full-trajectory sequences for degradation tracking.
2. UCI AI4I 2020 Tabular Datasets.
   - Stratified train/val splitting.
   - Class weight computation for imbalanced binary failure classification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CmapssSequenceDataset(Dataset):
    """PyTorch Dataset for NASA C-MAPSS temporal sequences."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        engine_ids: Optional[np.ndarray] = None,
        end_cycles: Optional[np.ndarray] = None,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.engine_ids = engine_ids
        self.end_cycles = end_cycles

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class Ai4iDataset(Dataset):
    """PyTorch Dataset for UCI AI4I tabular features."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_cmapss_sequence_split(
    subset: str = "FD001",
    window_length: int = 30,
    split: str = "train",
    base_dir: Union[str, Path] = Path("data/processed/nasa"),
) -> Dict[str, Any]:
    """Load a specific sequence NPZ archive."""
    file_path = Path(base_dir) / subset / f"sequences_w{window_length}_{split}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Sequence archive not found: {file_path}")

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


def get_terminal_test_indices(engine_ids: np.ndarray) -> np.ndarray:
    """Extract indices corresponding to the final observed window for each engine."""
    unique_engines = np.unique(engine_ids)
    terminal_indices = []
    for uid in unique_engines:
        idxs = np.where(engine_ids == uid)[0]
        terminal_indices.append(idxs[-1])
    return np.array(terminal_indices, dtype=np.int64)


def get_cmapss_dataloaders(
    subset: str = "FD001",
    window_length: int = 30,
    target_type: str = "rul",  # 'rul', 'ew10', 'ew20', 'ew30'
    batch_size: int = 256,
    val_batch_size: int = 256,
    base_dir: Union[str, Path] = Path("data/processed/nasa"),
) -> Dict[str, Any]:
    """Build train, val, and test DataLoaders for NASA C-MAPSS sequences."""
    train_data = load_cmapss_sequence_split(subset, window_length, "train", base_dir)
    val_data = load_cmapss_sequence_split(subset, window_length, "val", base_dir)
    test_data = load_cmapss_sequence_split(subset, window_length, "test", base_dir)

    target_key = f"y_{target_type}" if not target_type.startswith("y_") else target_type

    y_train = train_data[target_key]
    y_val = val_data[target_key]
    y_test = test_data[target_key]

    train_dataset = TensorDataset(torch.tensor(train_data["X"], dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data["X"], dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_data["X"], dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=False)

    terminal_test_indices = get_terminal_test_indices(test_data["engine_ids"])

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "terminal_test_indices": terminal_test_indices,
        "num_features": train_data["X"].shape[2],
        "sequence_length": window_length,
    }


def get_ai4i_dataloaders(
    base_dir: Union[str, Path] = Path("data/processed/ai4i"),
    val_ratio: float = 0.2,
    batch_size: int = 64,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build stratified train, val, and test DataLoaders for UCI AI4I."""
    base_dir = Path(base_dir)
    X_train_df = pd.read_csv(base_dir / "X_train.csv")
    y_train_df = pd.read_csv(base_dir / "y_train.csv")
    X_test_df = pd.read_csv(base_dir / "X_test.csv")
    y_test_df = pd.read_csv(base_dir / "y_test.csv")

    X_train_full = X_train_df.to_numpy(dtype=np.float32)
    y_train_full = y_train_df["Machine failure"].to_numpy(dtype=np.float32)

    X_test = X_test_df.to_numpy(dtype=np.float32)
    y_test = y_test_df["Machine failure"].to_numpy(dtype=np.float32)

    # Stratified split from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=random_state,
    )

    # Compute positive class weight
    num_neg = np.sum(y_tr == 0)
    num_pos = np.sum(y_tr == 1)
    pos_weight = float(num_neg / max(num_pos, 1))

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "input_dim": X_tr.shape[1],
        "pos_weight": pos_weight,
    }
