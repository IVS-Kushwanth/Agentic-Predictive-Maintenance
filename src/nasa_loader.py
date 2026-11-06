"""NASA C-MAPSS Dataset Loader Module.

Handles loading, column assigning, and parsing of NASA Turbofan Engine Degradation
Simulation datasets (FD001, FD002, FD003, FD004).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml


def get_cmapss_column_names(config: Optional[dict] = None) -> List[str]:
    """Return standard 26 column names for C-MAPSS data."""
    if config and "nasa" in config and "columns" in config["nasa"]:
        cols = config["nasa"]["columns"]
        return cols["index_columns"] + cols["setting_columns"] + cols["sensor_columns"]
    
    index_cols = ["unit_id", "cycle"]
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    return index_cols + setting_cols + sensor_cols


def load_nasa_raw_txt(file_path: Union[str, Path], col_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a raw whitespace-delimited NASA C-MAPSS text file into a DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NASA file not found: {file_path}")

    if col_names is None:
        col_names = get_cmapss_column_names()

    # Load with whitespace delimiter; drop trailing NaN columns if present
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="c")
    
    # If the file had trailing spaces, there might be more columns than expected
    if df.shape[1] > len(col_names):
        # Drop columns beyond the expected 26 if they are all NaN
        extra_cols = df.columns[len(col_names):]
        if df[extra_cols].isna().all().all():
            df = df.iloc[:, :len(col_names)]
        else:
            raise ValueError(
                f"File {file_path.name} has {df.shape[1]} columns, expected {len(col_names)}."
            )
    elif df.shape[1] < len(col_names):
        raise ValueError(
            f"File {file_path.name} has {df.shape[1]} columns, expected at least {len(col_names)}."
        )

    df.columns = col_names
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    
    # Ensure float64 for all numerical measurements
    float_cols = [c for c in df.columns if c not in ["unit_id", "cycle"]]
    df[float_cols] = df[float_cols].astype(np.float64)

    return df


def load_nasa_test_rul(file_path: Union[str, Path]) -> np.ndarray:
    """Load official test RUL ground-truth vector from RUL_FD00X.txt."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"RUL file not found: {file_path}")

    rul_values = pd.read_csv(file_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    return rul_values


def load_nasa_subset(
    subset: str,
    raw_dir: Union[str, Path] = "data/raw/nasa_cmapss"
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and official test RUL for a specific subset (e.g. 'FD001')."""
    raw_path = Path(raw_dir)
    train_file = raw_path / f"train_{subset}.txt"
    test_file = raw_path / f"test_{subset}.txt"
    rul_file = raw_path / f"RUL_{subset}.txt"

    train_df = load_nasa_raw_txt(train_file)
    test_df = load_nasa_raw_txt(test_file)
    test_rul = load_nasa_test_rul(rul_file)

    return train_df, test_df, test_rul


if __name__ == "__main__":
    print("Testing NASA Loader...")
    raw_dir = Path("data/raw/nasa_cmapss")
    if not raw_dir.exists():
        raw_dir = Path("agentic-predictive-maintenance/data/raw/nasa_cmapss")
    train, test, rul = load_nasa_subset("FD001", raw_dir=raw_dir)
    print(f"FD001 Train shape: {train.shape}, Engines: {train['unit_id'].nunique()}")
    print(f"FD001 Test shape:  {test.shape}, Engines: {test['unit_id'].nunique()}")
    print(f"FD001 Test RUL shape: {rul.shape}")
