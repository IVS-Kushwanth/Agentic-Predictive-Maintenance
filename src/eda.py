"""Exploratory Data Analysis (EDA) and Visualization Module.

Generates publication-quality figures for:
1. NASA C-MAPSS:
   - Operating cycle distributions
   - Cycles per engine
   - Sensor measurements over operational cycles
   - Individual engine degradation trajectories
   - Sensor correlation heatmaps
   - Operational setting distributions
   - Sensor distributions (boxplots)
   - Sensor variability analysis
   - Cross-engine sensor behavior comparisons
   - Cross-subset comparisons (FD001 vs FD002 vs FD003 vs FD004)
   - RUL distribution across train and test
   - Degradation progression normalized by life percentage
   - Sensor behavior grouped by RUL health regimes (Healthy, Degraded, Critical)

2. UCI AI4I 2020:
   - Target class distribution
   - Failure mode breakdown
   - Feature distributions by machine failure status
   - Correlation heatmap with target
   - Physical feature relationships (Mechanical Power, Temp Difference, Torque)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend for headless generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nasa_loader import load_nasa_subset
from rul_generation import compute_training_rul, reconstruct_test_rul
from ai4i_loader import load_ai4i_raw
from feature_engineering import engineer_ai4i_features


# Set consistent style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def plot_nasa_operating_cycles(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 1: Engine Operating Cycles Distribution."""
    max_cycles = train_df.groupby("unit_id")["cycle"].max()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(max_cycles, kde=True, bins=25, color="#1f77b4", ax=ax)
    ax.axvline(max_cycles.mean(), color="red", linestyle="--", label=f"Mean: {max_cycles.mean():.1f}")
    ax.axvline(max_cycles.median(), color="green", linestyle=":", label=f"Median: {max_cycles.median():.1f}")
    ax.set_title("NASA C-MAPSS FD001: Distribution of Engine Maximum Operating Cycles", fontsize=12, pad=10)
    ax.set_xlabel("Max Cycles to Failure (Engine Lifetime)")
    ax.set_ylabel("Engine Count")
    ax.legend()

    out_file = out_dir / "01_operating_cycles_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_cycles_per_engine(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 2: Number of Cycles Per Engine (Sorted Bar Chart)."""
    engine_lifetimes = train_df.groupby("unit_id")["cycle"].max().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(engine_lifetimes)), engine_lifetimes["cycle"], color="#2ca02c", alpha=0.85, width=0.8)
    ax.axhline(engine_lifetimes["cycle"].mean(), color="red", linestyle="--", label=f"Mean Lifetime ({engine_lifetimes['cycle'].mean():.1f} cycles)")
    ax.set_title("NASA C-MAPSS FD001: Operational Lifetime Per Engine (Sorted)", fontsize=12, pad=10)
    ax.set_xlabel("Engine Rank (Sorted by Lifetime)")
    ax.set_ylabel("Total Operational Cycles")
    ax.set_xlim(-1, len(engine_lifetimes))
    ax.legend()

    out_file = out_dir / "02_cycles_per_engine.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_sensors_over_cycles(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 3: Sensor Measurements Over Operational Cycles (Sample Engine)."""
    # Pick engine 1
    engine1 = train_df[train_df["unit_id"] == 1]
    key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, s in enumerate(key_sensors):
        ax = axes[idx]
        ax.plot(engine1["cycle"], engine1[s], color="#d62728", lw=1.5)
        ax.set_title(f"Engine 1: {s} Degradation Profile", fontsize=11)
        ax.set_ylabel("Reading")
        if idx >= 4:
            ax.set_xlabel("Operational Cycle")

    fig.suptitle("Sensor Trajectories Over Life Cycle (Engine 1)", fontsize=13, y=0.99)
    plt.tight_layout()

    out_file = out_dir / "03_sensors_over_cycles.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_individual_trajectories(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 4: Individual Engine Degradation Trajectories for Multiple Engines."""
    # Engine 1, 2, 3
    sample_engines = [1, 2, 3]
    sensor = "sensor_11"  # Static Pressure at HPC outlet

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for uid, c in zip(sample_engines, colors):
        sub = train_df[train_df["unit_id"] == uid]
        ax.plot(sub["cycle"], sub[sensor], label=f"Engine {uid} (Failed at cycle {sub['cycle'].max()})", color=c, lw=1.8)

    ax.set_title(f"Individual Engine Degradation Trajectories ({sensor})", fontsize=12, pad=10)
    ax.set_xlabel("Operational Cycle")
    ax.set_ylabel(f"{sensor} Measurement")
    ax.legend()

    out_file = out_dir / "04_individual_engine_trajectories.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_correlation_heatmap(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 5: Sensor Correlation Heatmap."""
    # Exclude constant sensors
    from data_cleaning import detect_constant_columns
    const_cols = detect_constant_columns(train_df)
    active_cols = [c for c in train_df.columns if c not in const_cols and c != "unit_id"]

    corr = train_df[active_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("NASA C-MAPSS FD001: Active Features Correlation Heatmap (including RUL)", fontsize=13, pad=12)

    out_file = out_dir / "05_sensor_correlation_heatmap.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_operational_settings(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 6: Operational Setting Distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    settings = ["setting_1", "setting_2", "setting_3"]

    for idx, s in enumerate(settings):
        ax = axes[idx]
        sns.histplot(train_df[s], kde=(train_df[s].nunique() > 10), ax=ax, color="#9467bd", bins=20)
        ax.set_title(f"Distribution of {s}", fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.suptitle("NASA C-MAPSS FD001: Operational Settings Profile (Sea Level)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_file = out_dir / "06_operational_setting_distributions.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_sensor_distributions(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 7: Sensor Distributions (Boxplots)."""
    # Active sensors
    active_sensors = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15"
    ]
    
    # Standardize temporarily for shared boxplot visualization
    norm_df = (train_df[active_sensors] - train_df[active_sensors].mean()) / train_df[active_sensors].std()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=norm_df, palette="Set2", ax=ax, showfliers=False)
    ax.set_title("NASA C-MAPSS FD001: Standardized Sensor Distributions", fontsize=12, pad=10)
    ax.set_ylabel("Z-score Normalized Value")
    ax.set_xticks(range(len(active_sensors)))
    ax.set_xticklabels(active_sensors, rotation=35, ha="right")

    out_file = out_dir / "07_sensor_distributions.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_sensor_variability(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 8: Sensor Variability Analysis (Comparing Std Dev)."""
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = train_df[sensor_cols].std().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(stds)), stds.values, color="#e377c2", width=0.7)
    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels(stds.index, rotation=45, ha="right")
    ax.set_title("NASA C-MAPSS FD001: Raw Sensor Measurement Standard Deviations", fontsize=12, pad=10)
    ax.set_ylabel("Standard Deviation")
    
    # Highlight zero variance sensors
    for i, bar in enumerate(bars):
        if stds.values[i] < 1e-4:
            bar.set_color("#7f7f7f")

    out_file = out_dir / "08_sensor_variability_analysis.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_cross_engine_behavior(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 9: Sensor Behavior Across Different Engines (Overlay)."""
    engines = [1, 5, 10, 20, 50]
    sensor = "sensor_4"  # LPT coolant bleed

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("tab10", len(engines))

    for idx, e in enumerate(engines):
        sub = train_df[train_df["unit_id"] == e]
        ax.plot(sub["cycle"], sub[sensor], label=f"Engine {e}", color=palette[idx], alpha=0.85, lw=1.5)

    ax.set_title(f"Cross-Engine Sensor Behavior Comparison ({sensor})", fontsize=12, pad=10)
    ax.set_xlabel("Operational Cycle")
    ax.set_ylabel("Sensor Reading")
    ax.legend(title="Engine ID")

    out_file = out_dir / "09_cross_engine_behavior.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_subset_comparison(raw_dir: Path, out_dir: Path) -> Path:
    """Plot 10: Comparison of FD001, FD002, FD003, FD004."""
    subset_info = {
        "FD001": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 1, "records": 20631},
        "FD002": {"train_engines": 260, "test_engines": 259, "conditions": 6, "fault_modes": 1, "records": 53759},
        "FD003": {"train_engines": 100, "test_engines": 100, "conditions": 1, "fault_modes": 2, "records": 24720},
        "FD004": {"train_engines": 248, "test_engines": 249, "conditions": 6, "fault_modes": 2, "records": 61249},
    }
    comp_df = pd.DataFrame(subset_info).T

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # 1. Total Records
    sns.barplot(data=comp_df, x=comp_df.index, y="records", hue=comp_df.index, palette="Blues_d", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Training Records Volume")
    axes[0, 0].set_ylabel("Records")

    # 2. Train Engines
    sns.barplot(data=comp_df, x=comp_df.index, y="train_engines", hue=comp_df.index, palette="Greens_d", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Fleet Size (Train Engines)")
    axes[0, 1].set_ylabel("Engines")

    # 3. Operating Conditions
    sns.barplot(data=comp_df, x=comp_df.index, y="conditions", hue=comp_df.index, palette="Oranges_d", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Operating Conditions (Regimes)")
    axes[1, 0].set_ylabel("Distinct Conditions")

    # 4. Fault Modes
    sns.barplot(data=comp_df, x=comp_df.index, y="fault_modes", hue=comp_df.index, palette="Reds_d", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Simulated Degradation Modes")
    axes[1, 1].set_ylabel("Fault Modes Count")

    fig.suptitle("Architectural Comparison: NASA C-MAPSS Subsets (FD001-FD004)", fontsize=13, y=1.01)
    plt.tight_layout()

    out_file = out_dir / "10_subset_comparison.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_rul_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 11: RUL Distribution (Train vs Reconstructed Test)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(train_df["RUL"], label="Train RUL (Run-to-failure)", color="#1f77b4", lw=2, ax=ax, fill=True, alpha=0.3)
    sns.kdeplot(test_df["RUL"], label="Test RUL (Reconstructed Trajectories)", color="#ff7f0e", lw=2, ax=ax, fill=True, alpha=0.3)
    ax.set_title("NASA C-MAPSS FD001: RUL Density Distribution", fontsize=12, pad=10)
    ax.set_xlabel("Remaining Useful Life (Cycles)")
    ax.set_ylabel("Density")
    ax.legend()

    out_file = out_dir / "11_rul_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_degradation_progression(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 12: Engine Degradation Progression (Normalized Life %)."""
    df = train_df.copy()
    max_c = df.groupby("unit_id")["cycle"].transform("max")
    df["life_percent"] = ((df["cycle"] / max_c) * 100.0).round().astype(int)

    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()

    for idx, s in enumerate(sensors):
        ax = axes[idx]
        sns.lineplot(data=df, x="life_percent", y=s, ax=ax, color="#2b5c8f", errorbar="sd")
        ax.set_title(f"Fleet Degradation Curve: {s}", fontsize=11)
        ax.set_ylabel("Measurement")
        if idx >= 2:
            ax.set_xlabel("Engine Life Completed (%)")

    fig.suptitle("Fleet Mean Degradation Progression with Standard Deviation Band", fontsize=13, y=1.01)
    plt.tight_layout()

    out_file = out_dir / "12_degradation_progression.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_nasa_sensor_rul_regimes(train_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 13: Sensor Behavior Grouped by RUL Ranges."""
    df = train_df.copy()
    bins = [-1, 30, 100, 1000]
    labels = ["Critical (RUL <= 30)", "Degraded (30 < RUL <= 100)", "Healthy (RUL > 100)"]
    df["Health_Regime"] = pd.cut(df["RUL"], bins=bins, labels=labels)

    active_sensors = ["sensor_2", "sensor_4", "sensor_11", "sensor_12"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, s in enumerate(active_sensors):
        ax = axes[idx]
        sns.boxplot(data=df, x="Health_Regime", y=s, hue="Health_Regime", palette=["#d62728", "#ff7f0e", "#2ca02c"], legend=False, ax=ax, showfliers=False)
        ax.set_title(f"{s} Across Health Regimes", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Measurement")

    fig.suptitle("NASA C-MAPSS FD001: Sensor Profiles Across RUL Health Regimes", fontsize=13, y=1.01)
    plt.tight_layout()

    out_file = out_dir / "13_sensor_behavior_rul_ranges.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def generate_all_nasa_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_dir: Path,
    out_dir: Path = Path("reports/figures/nasa")
) -> List[Path]:
    """Execute generation of all 13 required NASA plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        plot_nasa_operating_cycles(train_df, out_dir),
        plot_nasa_cycles_per_engine(train_df, out_dir),
        plot_nasa_sensors_over_cycles(train_df, out_dir),
        plot_nasa_individual_trajectories(train_df, out_dir),
        plot_nasa_correlation_heatmap(train_df, out_dir),
        plot_nasa_operational_settings(train_df, out_dir),
        plot_nasa_sensor_distributions(train_df, out_dir),
        plot_nasa_sensor_variability(train_df, out_dir),
        plot_nasa_cross_engine_behavior(train_df, out_dir),
        plot_nasa_subset_comparison(raw_dir, out_dir),
        plot_nasa_rul_distribution(train_df, test_df, out_dir),
        plot_nasa_degradation_progression(train_df, out_dir),
        plot_nasa_sensor_rul_regimes(train_df, out_dir),
    ]
    return plots


def generate_all_ai4i_plots(
    df_raw: pd.DataFrame,
    out_dir: Path = Path("reports/figures/ai4i")
) -> List[Path]:
    """Generate comprehensive EDA visualizations for UCI AI4I 2020."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df_eng, _ = engineer_ai4i_features(df_raw)

    generated = []

    # 1. Target Class Balance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = df_raw["Machine failure"].value_counts()
    axes[0].pie(counts, labels=["No Failure (0)", "Failure (1)"], autopct="%1.2f%%", colors=["#2ca02c", "#d62728"], startangle=90, explode=(0, 0.15))
    axes[0].set_title("Machine Failure Class Ratio")

    sns.countplot(data=df_raw, x="Type", hue="Machine failure", palette=["#2ca02c", "#d62728"], ax=axes[1])
    axes[1].set_title("Failure Distribution by Product Type (L/M/H)")
    axes[1].set_ylabel("Count")
    fig.suptitle("UCI AI4I 2020: Target Imbalance Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    p1 = out_dir / "01_target_class_balance.png"
    fig.savefig(p1)
    plt.close(fig)
    generated.append(p1)

    # 2. Failure Modes Breakdown
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_counts = df_raw[modes].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=mode_counts.index, y=mode_counts.values, hue=mode_counts.index, palette="rocket", legend=False, ax=ax)
    ax.set_title("UCI AI4I 2020: Specific Failure Mode Frequencies", fontsize=12, pad=10)
    ax.set_ylabel("Occurrence Count")
    ax.set_xlabel("Failure Mode Code")
    for i, v in enumerate(mode_counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    p2 = out_dir / "02_failure_modes_breakdown.png"
    fig.savefig(p2)
    plt.close(fig)
    generated.append(p2)

    # 3. Feature Distributions by Failure Status
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    features = ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Temp difference [K]"]
    for idx, f in enumerate(features):
        ax = axes.flatten()[idx]
        sns.boxplot(data=df_eng, x="Machine failure", y=f, hue="Machine failure", palette=["#2ca02c", "#d62728"], legend=False, ax=ax, showfliers=False)
        ax.set_title(f"{f} by Failure Status", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal (0)", "Failure (1)"])
        ax.set_xlabel("")
    fig.suptitle("Physical Feature Distributions Conditioned on Machine Failure", fontsize=13, y=1.01)
    plt.tight_layout()
    p3 = out_dir / "03_features_by_failure_status.png"
    fig.savefig(p3)
    plt.close(fig)
    generated.append(p3)

    # 4. Correlation Heatmap
    corr_cols = [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]", "Mechanical power [W]", "Temp difference [K]",
        "Overstrain indicator [Nm*min]", "Machine failure"
    ]
    corr = df_eng[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, annot_kws={"size": 8})
    ax.set_title("UCI AI4I 2020: Correlation Heatmap with Engineered Features", fontsize=12, pad=10)
    p4 = out_dir / "04_ai4i_correlation_heatmap.png"
    fig.savefig(p4)
    plt.close(fig)
    generated.append(p4)

    # 5. Physical Mechanics: Power vs Torque vs Speed
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df_eng["Rotational speed [rpm]"],
        df_eng["Torque [Nm]"],
        c=df_eng["Mechanical power [W]"],
        cmap="viridis",
        alpha=0.6,
        s=15
    )
    plt.colorbar(scatter, ax=ax, label="Mechanical Power [W]")
    ax.set_title("Machine Operating Space: Torque vs Rotational Speed", fontsize=12, pad=10)
    ax.set_xlabel("Rotational Speed [rpm]")
    ax.set_ylabel("Torque [Nm]")
    p5 = out_dir / "05_physical_mechanics_power.png"
    fig.savefig(p5)
    plt.close(fig)
    generated.append(p5)

    return generated


def main():
    """Run all EDA visualizations."""
    raw_nasa = Path("data/raw/nasa_cmapss")
    raw_ai4i = Path("data/raw/ai4i/ai4i2020.csv")

    print("Loading data for EDA...")
    train_raw, test_raw, test_rul_vec = load_nasa_subset("FD001", raw_dir=raw_nasa)
    train_labeled = compute_training_rul(train_raw)
    test_labeled = reconstruct_test_rul(test_raw, test_rul_vec)

    print("Generating NASA C-MAPSS figures...")
    nasa_figs = generate_all_nasa_plots(train_labeled, test_labeled, raw_nasa)
    print(f"Generated {len(nasa_figs)} NASA figures in reports/figures/nasa/")

    print("Generating UCI AI4I figures...")
    ai4i_df = load_ai4i_raw(raw_ai4i)
    ai4i_figs = generate_all_ai4i_plots(ai4i_df)
    print(f"Generated {len(ai4i_figs)} AI4I figures in reports/figures/ai4i/")


if __name__ == "__main__":
    main()
