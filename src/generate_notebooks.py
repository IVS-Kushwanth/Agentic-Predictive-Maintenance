"""Notebook Generation Helper.

Programmatically generates the 5 required Jupyter notebooks for Phase 1:
1. 01_dataset_acquisition.ipynb
2. 02_nasa_dataset_eda.ipynb
3. 03_nasa_temporal_preprocessing.ipynb
4. 04_ai4i_eda.ipynb
5. 05_ai4i_preprocessing.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def build_notebook_1():
    cells = [
        md_cell("""# 01 - Dataset Acquisition & Verification
### Agentic Predictive Maintenance Framework — Phase 1

This notebook demonstrates the authentic acquisition, extraction, and verification of:
1. **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset** (FD001–FD004)
2. **UCI AI4I 2020 Predictive Maintenance Dataset**"""),
        code_cell("""import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from data_download import download_nasa_cmapss, download_ai4i, load_config
from data_validation import validate_nasa_dataset, validate_ai4i_dataset"""),
        md_cell("## 1. Load Project Configuration"),
        code_cell("""config = load_config(str(project_root / "config" / "config.yaml"))
print(f"Project: {config['project']['name']} v{config['project']['version']}")
print(f"NASA Raw Directory: {config['paths']['raw_nasa_dir']}")
print(f"AI4I Raw Directory: {config['paths']['raw_ai4i_dir']}")"""),
        md_cell("## 2. Verify NASA C-MAPSS Files"),
        code_cell("""nasa_dir = project_root / config["paths"]["raw_nasa_dir"]
expected_nasa_files = [f"{split}_{subset}.txt" for subset in config["nasa"]["subsets"] for split in ["train", "test", "RUL"]]
file_status = {f: (nasa_dir / f).exists() for f in expected_nasa_files}
df_status = pd.DataFrame(list(file_status.items()), columns=["File", "Exists"])
print(f"All {len(expected_nasa_files)} NASA files verified: {df_status['Exists'].all()}")
display(df_status)"""),
        md_cell("## 3. Verify UCI AI4I 2020 File"),
        code_cell("""ai4i_file = project_root / config["paths"]["raw_ai4i_dir"] / config["ai4i"]["raw_file"]
print(f"AI4I file exists: {ai4i_file.exists()} ({ai4i_file.stat().st_size / 1024:.2f} KB)")"""),
    ]
    return make_notebook(cells)


def build_notebook_2():
    cells = [
        md_cell("""# 02 - NASA C-MAPSS Exploratory Data Analysis (EDA)
### Turbofan Engine Degradation Trajectories & Multi-Subset Analysis"""),
        code_cell("""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from nasa_loader import load_nasa_subset
from rul_generation import compute_training_rul, reconstruct_test_rul, compute_rul_statistics
from eda import generate_all_nasa_plots"""),
        md_cell("## 1. Load FD001 Benchmark Dataset"),
        code_cell("""raw_dir = project_root / "data" / "raw" / "nasa_cmapss"
train_df, test_df, test_rul = load_nasa_subset("FD001", raw_dir=raw_dir)

train_labeled = compute_training_rul(train_df)
test_labeled = reconstruct_test_rul(test_df, test_rul)

print(f"FD001 Train Records: {len(train_labeled)}, Engines: {train_labeled['unit_id'].nunique()}")
print(f"FD001 Test Records:  {len(test_labeled)}, Engines: {test_labeled['unit_id'].nunique()}")"""),
        md_cell("## 2. Summary Statistics for RUL"),
        code_cell("""train_stats = compute_rul_statistics(train_labeled, "Train FD001")
test_stats = compute_rul_statistics(test_labeled, "Test FD001")
display(pd.DataFrame([train_stats, test_stats]))"""),
        md_cell("## 3. Visualizations Generated"),
        code_cell("""figs_dir = project_root / "reports" / "figures" / "nasa"
print("Generated NASA Figures:")
for fig_path in sorted(figs_dir.glob("*.png")):
    print(f" - {fig_path.name}")"""),
    ]
    return make_notebook(cells)


def build_notebook_3():
    cells = [
        md_cell("""# 03 - NASA Temporal Preprocessing & Sequence Construction
### Feature Engineering, Scaler Fitting, and 3D Sliding Window NPZ Storage"""),
        code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from sequence_generation import load_sequences_npz"""),
        md_cell("## 1. Inspect Scaled Tabular Data"),
        code_cell("""proc_dir = project_root / "data" / "processed" / "nasa" / "FD001"
train_tab = pd.read_csv(proc_dir / "train_tabular.csv")
val_tab = pd.read_csv(proc_dir / "val_tabular.csv")
test_tab = pd.read_csv(proc_dir / "test_tabular.csv")

print(f"Train Tabular Shape: {train_tab.shape}")
print(f"Val Tabular Shape:   {val_tab.shape}")
print(f"Test Tabular Shape:  {test_tab.shape}")"""),
        md_cell("## 2. Inspect 3D Temporal Sequence NPZ Files"),
        code_cell("""for w in [30, 50, 60]:
    seq_train = load_sequences_npz(proc_dir / f"sequences_w{w}_train.npz")
    seq_val = load_sequences_npz(proc_dir / f"sequences_w{w}_val.npz")
    seq_test = load_sequences_npz(proc_dir / f"sequences_w{w}_test.npz")
    
    print(f"\\n--- Window Length: {w} ---")
    print(f"Train X: {seq_train['X'].shape}, y_rul: {seq_train['y_rul'].shape}")
    print(f"Val X:   {seq_val['X'].shape}, y_rul: {seq_val['y_rul'].shape}")
    print(f"Test X:  {seq_test['X'].shape}, y_rul: {seq_test['y_rul'].shape}")"""),
        md_cell("## 3. Data Leakage and Boundary Verification Checklist"),
        code_cell("""# Verify zero engine overlap
train_engines = set(train_tab['unit_id'].unique())
val_engines = set(val_tab['unit_id'].unique())
assert len(train_engines.intersection(val_engines)) == 0, "Engine overlap detected!"
print(f"Verified: Zero engine overlap between Train ({len(train_engines)} engines) and Val ({len(val_engines)} engines).")"""),
    ]
    return make_notebook(cells)


def build_notebook_4():
    cells = [
        md_cell("""# 04 - UCI AI4I 2020 Exploratory Data Analysis (EDA)
### Industrial Machine Failure Analysis, Failure Modes, and Class Imbalance"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from ai4i_loader import load_ai4i_raw
from feature_engineering import engineer_ai4i_features"""),
        md_cell("## 1. Load Raw AI4I Dataset"),
        code_cell("""raw_file = project_root / "data" / "raw" / "ai4i" / "ai4i2020.csv"
df = load_ai4i_raw(raw_file)
print(f"AI4I Shape: {df.shape}")
print(f"Target distribution:\\n{df['Machine failure'].value_counts()}")
print(f"Failure rate: {df['Machine failure'].mean() * 100:.2f}%")"""),
        md_cell("## 2. Failure Modes Decomposition"),
        code_cell("""modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
display(df[modes].sum().to_frame(name="Count"))"""),
    ]
    return make_notebook(cells)


def build_notebook_5():
    cells = [
        md_cell("""# 05 - UCI AI4I 2020 Preprocessing & Stratified Splitting
### Physical Feature Engineering, ColumnTransformer Pipeline, and Saved Artifacts"""),
        code_cell("""import sys
from pathlib import Path
import joblib
import pandas as pd

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))"""),
        md_cell("## 1. Inspect Processed AI4I Artifacts"),
        code_cell("""proc_dir = project_root / "data" / "processed" / "ai4i"
X_train = pd.read_csv(proc_dir / "X_train.csv")
X_test = pd.read_csv(proc_dir / "X_test.csv")
y_train = pd.read_csv(proc_dir / "y_train.csv")
y_test = pd.read_csv(proc_dir / "y_test.csv")

print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_test Shape:  {X_test.shape}, y_test Shape:  {y_test.shape}")
print(f"Train Failure Rate: {y_train['Machine failure'].mean():.4f}")
print(f"Test Failure Rate:  {y_test['Machine failure'].mean():.4f}")"""),
        md_cell("## 2. Verify Saved Fitted Preprocessing Pipeline"),
        code_cell("""pipeline_path = project_root / "models" / "preprocessing" / "ai4i_preprocessor.joblib"
pipeline = joblib.load(pipeline_path)
print(f"Loaded Preprocessing Pipeline: {pipeline}")"""),
    ]
    return make_notebook(cells)


def main():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01_dataset_acquisition.ipynb": build_notebook_1(),
        "02_nasa_dataset_eda.ipynb": build_notebook_2(),
        "03_nasa_temporal_preprocessing.ipynb": build_notebook_3(),
        "04_ai4i_eda.ipynb": build_notebook_4(),
        "05_ai4i_preprocessing.ipynb": build_notebook_5(),
    }

    for name, nb in notebooks.items():
        file_path = nb_dir / name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Created notebook: {file_path}")


if __name__ == "__main__":
    main()
