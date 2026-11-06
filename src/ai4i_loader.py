"""UCI AI4I 2020 Predictive Maintenance Dataset Loader Module.

Handles loading, schema validation, and parsing of the AI4I 2020 dataset.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def load_ai4i_raw(file_path: Union[str, Path] = "data/raw/ai4i/ai4i2020.csv") -> pd.DataFrame:
    """Load the raw UCI AI4I 2020 CSV dataset."""
    file_path = Path(file_path)
    if not file_path.exists():
        alt_path = Path("agentic-predictive-maintenance") / file_path
        if alt_path.exists():
            file_path = alt_path
        else:
            raise FileNotFoundError(f"AI4I dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Standard expected columns
    expected_cols = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"AI4I dataset is missing expected columns: {missing_cols}")

    # Set appropriate types
    df["UDI"] = df["UDI"].astype(int)
    df["Product ID"] = df["Product ID"].astype(str)
    df["Type"] = df["Type"].astype(str)
    df["Air temperature [K]"] = df["Air temperature [K]"].astype(np.float64)
    df["Process temperature [K]"] = df["Process temperature [K]"].astype(np.float64)
    df["Rotational speed [rpm]"] = df["Rotational speed [rpm]"].astype(np.float64)
    df["Torque [Nm]"] = df["Torque [Nm]"].astype(np.float64)
    df["Tool wear [min]"] = df["Tool wear [min]"].astype(np.float64)
    
    # Target and failure modes
    for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = df[col].astype(int)

    return df


if __name__ == "__main__":
    print("Testing AI4I Loader...")
    df_ai4i = load_ai4i_raw("data/raw/ai4i/ai4i2020.csv")
    print(f"AI4I shape: {df_ai4i.shape}")
    print(f"Machine failure distribution:\n{df_ai4i['Machine failure'].value_counts(normalize=True)}")
