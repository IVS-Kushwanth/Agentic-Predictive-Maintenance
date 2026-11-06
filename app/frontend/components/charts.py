"""
Interactive Plotly Visualizations for Streamlit Frontend.
Provides industrial-grade sensor trend lines, breakdown gauges,
SHAP feature attribution charts, and early-warning horizon distributions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_sensor_trends(sequence_2d: np.ndarray, feature_names: List[str], selected_sensors: Optional[List[str]] = None) -> go.Figure:
    """
    Plots temporal degradation trajectories across operational flight cycles for NASA turbofans.
    """
    fig = go.Figure()
    n_cycles = sequence_2d.shape[0]
    cycles = list(range(1, n_cycles + 1))

    # Pick top sensors to display if not specified
    if not selected_sensors:
        # Default to high-wear core sensors
        candidate_indices = [3, 4, 8, 11, 13]  # e.g., T30, LPT exhaust, HPC coolant
        sensor_indices = [i for i in candidate_indices if i < sequence_2d.shape[1]]
    else:
        sensor_indices = [feature_names.index(s) for s in selected_sensors if s in feature_names]

    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]

    for idx, s_idx in enumerate(sensor_indices):
        s_name = feature_names[s_idx]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=cycles,
            y=sequence_2d[:, s_idx],
            mode="lines+markers",
            name=s_name,
            line=dict(width=2.5, color=color),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="Multivariate Sensor Degradation Profile Across Operating Window",
        xaxis_title="Operating Cycles within Window (1 to 30)",
        yaxis_title="Normalized Sensor Telemetry",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", size=12, color="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_gauge_meter(
    value: float,
    title: str,
    max_val: float = 1.0,
    thresholds: Optional[List[float]] = None,
    unit: str = "%"
) -> go.Figure:
    """
    Renders an industrial radial gauge meter with color-coded safety zones.
    """
    th = thresholds or [0.4, 0.75]
    display_val = value * 100 if unit == "%" and max_val == 1.0 else value
    max_display = max_val * 100 if unit == "%" and max_val == 1.0 else max_val

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"suffix": f" {unit}", "font": {"size": 28, "color": "#F8FAFC"}},
        title={"text": title, "font": {"size": 15, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, max_display], "tickcolor": "#64748B"},
            "bar": {"color": "#3B82F6"},
            "bgcolor": "#1E293B",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, th[0] * (100 if unit == "%" and max_val == 1.0 else 1)], "color": "#064E3B"},
                {"range": [th[0] * (100 if unit == "%" and max_val == 1.0 else 1), th[1] * (100 if unit == "%" and max_val == 1.0 else 1)], "color": "#78350F"},
                {"range": [th[1] * (100 if unit == "%" and max_val == 1.0 else 1), max_display], "color": "#7F1D1D"}
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 4},
                "thickness": 0.8,
                "value": display_val
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        height=220,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_shap_attributions(important_features: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders horizontal waterfall / contribution bar chart of SHAP feature importance.
    """
    if not important_features:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A")
        return fig

    # Reverse order so highest importance appears at the top
    sorted_features = list(reversed(important_features[:10]))
    names = [f.get("feature", "feat") for f in sorted_features]
    vals = [f.get("importance", 0.0) for f in sorted_features]
    colors = ["#EF4444" if v >= 0 else "#10B981" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#334155", width=1))
    ))

    fig.update_layout(
        title="Top Explainable AI Feature Drivers (TreeSHAP Value Impact)",
        xaxis_title="Impact on Model Output (+ Increases Risk/Degradation, - Protects)",
        yaxis_title="Telemetry Feature / Operating Variable",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", size=11, color="#E2E8F0"),
        height=320,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_early_warning_distribution(ew_horizons: Dict[str, float]) -> go.Figure:
    """
    Renders bar chart of early-warning failure probabilities across horizons (10, 20, 30 cycles).
    """
    horizons = ["10 Cycles", "20 Cycles", "30 Cycles"]
    probs = [
        ew_horizons.get("risk_within_10_cycles", 0.0) * 100,
        ew_horizons.get("risk_within_20_cycles", 0.0) * 100,
        ew_horizons.get("risk_within_30_cycles", 0.0) * 100
    ]

    fig = go.Figure(go.Bar(
        x=horizons,
        y=probs,
        marker_color=["#EF4444" if p > 75 else "#F59E0B" if p > 40 else "#10B981" for p in probs],
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto"
    ))

    fig.update_layout(
        title="Early-Warning Horizon Breakdown",
        xaxis_title="Proximity Horizon",
        yaxis_title="Failure Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=240,
        margin=dict(l=30, r=30, t=40, b=30)
    )
    return fig


def plot_ai4i_operating_radar(features_dict: Dict[str, float]) -> go.Figure:
    """
    Renders radar chart of CNC operational stress variables.
    """
    categories = list(features_dict.keys())[:7]
    values = [features_dict[c] for c in categories]

    # Normalize roughly for radar visual comparison
    norm_vals = []
    for v in values:
        norm_vals.append(min(1.0, max(0.0, float(v) / 100.0 if float(v) > 1.0 else float(v))))

    # Close the radar loop
    categories.append(categories[0])
    norm_vals.append(norm_vals[0])

    fig = go.Figure(go.Scatterpolar(
        r=norm_vals,
        theta=categories,
        fill="toself",
        line=dict(color="#3B82F6", width=2),
        fillcolor="rgba(59, 130, 246, 0.25)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color="#64748B"),
            bgcolor="#1E293B"
        ),
        title="CNC Operating Stress Distribution",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        height=280,
        margin=dict(l=30, r=30, t=40, b=30)
    )
    return fig
