"""
Page 2: Machine Monitoring.
Displays real-time sensor measurements, temporal degradation curves (NASA),
and operating-condition stress profiles (AI4I) without false time-series assumptions.
"""

import streamlit as st
import numpy as np
import pandas as pd
from ..components.data_loader import get_sample_telemetry
from ..components.charts import plot_sensor_trends, plot_ai4i_operating_radar


def render_monitoring(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader(f"🔍 Machine Telemetry & Degradation Monitoring — {selected_unit}")

    telemetry_data, meta = get_sample_telemetry(dataset_type, selected_unit)

    if dataset_type == "NASA":
        st.markdown(
            "**Temporal Telemetry Context:** Displaying a 30-cycle sliding operational window for turbofan "
            f"`{meta['unit_id']}`. Sensor channels illustrate thermodynamic and aerodynamic degradation trajectories."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Monitoring Window", f"{meta['window_size']} Operating Cycles")
        with c2:
            st.metric("Monitored Sensor Channels", f"{meta['num_features']} Features")
        with c3:
            st.metric("Ground-Truth RUL", f"{meta['actual_rul']:.1f} Cycles")

        st.markdown("#### 📈 Multivariate Temporal Sensor Degradation")
        # Multiselect for sensor channels
        features = meta["feature_names"]
        default_sensors = [f for f in ["sensor_11", "sensor_4", "sensor_12", "sensor_13", "sensor_8"] if f in features]
        selected_channels = st.multiselect("Select Sensor Channels to Overlay:", features[:20], default=default_sensors)

        fig_trend = plot_sensor_trends(telemetry_data, feature_names=features, selected_sensors=selected_channels)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("#### 📋 Current Cycle Sensor Matrix (Terminal Observation)")
        # Show last cycle tabular values
        terminal_values = telemetry_data[-1, :]
        df_sensors = pd.DataFrame({
            "Sensor Feature": features[:25],
            "Normalized Reading": terminal_values[:25],
            "Telemetry Subsystem": ["High-Pressure Compressor" if "11" in s else "Low-Pressure Turbine" if "4" in s else "Fan / Bypass Duct" for s in features[:25]]
        })
        st.dataframe(df_sensors, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            "**Operational Stress Profile:** Displaying machine tool operating conditions for "
            f"`{meta['machine_id']}`. Note: UCI AI4I 2020 consists of independent discrete machining records, "
            "not a continuous time-series trajectory."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Machine Subsystem", "CNC Milling Spindle")
        with c2:
            st.metric("Operating Variables", f"{meta['num_features']} Engineered Features")
        with c3:
            status_text = "Machine Failure" if meta['actual_failure'] == 1 else "Normal Operation"
            st.metric("Ground-Truth Status", status_text, delta="Overstrain / Wear" if meta['actual_failure'] == 1 else "Nominal", delta_color="inverse" if meta['actual_failure'] == 1 else "normal")

        col_radar, col_table = st.columns([1, 1])

        features = meta["feature_names"]
        feat_dict = {features[i]: float(telemetry_data[i]) for i in range(len(features))}

        with col_radar:
            st.markdown("#### 🎯 Operating Stress Radar")
            fig_radar = plot_ai4i_operating_radar(feat_dict)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_table:
            st.markdown("#### ⚙️ Physical Operating Measurements")
            df_ai4i = pd.DataFrame({
                "Parameter Name": list(feat_dict.keys()),
                "Recorded Value": list(feat_dict.values())
            })
            st.dataframe(df_ai4i, use_container_width=True, hide_index=True)
