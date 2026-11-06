"""
Data Loader & Sample Telemetry Provider for Streamlit Frontend.
Loads actual validation samples from Phase 1 processed data and validates CSV uploads.
Never fabricates synthetic readings or silently reorders schema columns.
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Cache dataset objects in memory for fast UI interaction
_NASA_NPZ: Optional[Dict[str, Any]] = None
_NASA_TABULAR: Optional[pd.DataFrame] = None
_AI4I_X: Optional[pd.DataFrame] = None
_AI4I_Y: Optional[pd.Series] = None
_AI4I_META: Optional[Dict[str, Any]] = None


def get_nasa_data():
    global _NASA_NPZ, _NASA_TABULAR
    if _NASA_NPZ is None:
        npz_path = os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz")
        _NASA_NPZ = np.load(npz_path)

        csv_path = os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "val_tabular.csv")
        if os.path.exists(csv_path):
            _NASA_TABULAR = pd.read_csv(csv_path)

    return _NASA_NPZ, _NASA_TABULAR


def get_ai4i_data():
    global _AI4I_X, _AI4I_Y, _AI4I_META
    if _AI4I_X is None:
        x_path = os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv")
        y_path = os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")
        meta_path = os.path.join(BASE_DIR, "data", "processed", "ai4i", "metadata.json")

        _AI4I_X = pd.read_csv(x_path)
        _AI4I_Y = pd.read_csv(y_path).squeeze()
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                _AI4I_META = json.load(f)
        else:
            _AI4I_META = {}

    return _AI4I_X, _AI4I_Y, _AI4I_META


def get_available_units(dataset_type: str) -> List[str]:
    """Returns list of selectable engine or machine IDs."""
    if dataset_type == "NASA":
        npz, tab = get_nasa_data()
        if tab is not None and "unit_id" in tab.columns:
            units = sorted(list(tab["unit_id"].unique()))
            return [f"Engine-FD001-{int(u):03d}" for u in units[:30]]
        return [f"Engine-FD001-{i:03d}" for i in range(1, 31)]
    else:
        X, y, _ = get_ai4i_data()
        return [f"CNC-Tool-{i:03d} ({'Failure' if y.iloc[i]==1 else 'Normal'})" for i in range(min(40, len(X)))]


def get_sample_telemetry(dataset_type: str, selected_unit: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieves the actual feature array and metadata for a selected machine.
    """
    if dataset_type == "NASA":
        npz, tab = get_nasa_data()
        # Extract engine index
        try:
            unit_idx = int(selected_unit.split("-")[-1]) - 1
        except Exception:
            unit_idx = 0

        unit_idx = max(0, min(unit_idx, len(npz["X"]) - 1))
        # 3D sequence window (30, 153)
        seq_3d = npz["X"][unit_idx]
        true_rul = float(npz["y_rul"][unit_idx])
        meta = {
            "unit_id": f"FD001-{unit_idx+1:03d}",
            "actual_rul": true_rul,
            "window_size": seq_3d.shape[0],
            "num_features": seq_3d.shape[1],
            "feature_names": list(npz["feature_names"])
        }
        return seq_3d, meta

    else:
        X, y, meta_dict = get_ai4i_data()
        try:
            idx = int(selected_unit.split("-")[2].split(" ")[0])
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(X) - 1))

        row_1d = X.iloc[idx].to_numpy(dtype=np.float32)
        true_failure = int(y.iloc[idx])
        meta = {
            "machine_id": f"CNC-Tool-{idx:03d}",
            "actual_failure": true_failure,
            "num_features": len(row_1d),
            "feature_names": list(X.columns)
        }
        return row_1d, meta


def validate_and_parse_csv(uploaded_file, dataset_type: str) -> Tuple[np.ndarray, List[str]]:
    """
    Validates uploaded CSV file against exact schema.
    Never silently rearranges or fabricates missing columns.
    """
    df = pd.read_csv(uploaded_file)

    if dataset_type == "NASA":
        npz, _ = get_nasa_data()
        expected_cols = list(npz["feature_names"])
        # Check matching columns
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Uploaded NASA CSV is missing {len(missing)} required feature columns! First 5 missing: {missing[:5]}"
            )
        ordered_df = df[expected_cols]
        return ordered_df.to_numpy(dtype=np.float32), expected_cols

    else:
        X, _, _ = get_ai4i_data()
        expected_cols = list(X.columns)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Uploaded AI4I CSV is missing {len(missing)} required feature columns! Missing: {missing}"
            )
        ordered_df = df[expected_cols]
        return ordered_df.to_numpy(dtype=np.float32), expected_cols
