"""
Page 4: Explainable AI (XAI).
Presents TreeSHAP feature attributions, individual prediction breakdowns,
and readable engineering explanations grounded in actual model outputs.
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from ..components.data_loader import get_sample_telemetry
from ..components.charts import plot_shap_attributions


def render_explainability(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader(f"🧠 Explainable AI Diagnostic Attributions — {selected_unit}")

    telemetry_data, meta = get_sample_telemetry(dataset_type, selected_unit)
    features_list = telemetry_data[-1, :].tolist() if telemetry_data.ndim >= 2 else telemetry_data.tolist()

    explain_res = None
    with st.spinner("Calculating TreeSHAP Shapley values and feature importance weights..."):
        try:
            r = requests.post(
                f"{api_url}/api/explain",
                json={
                    "machine_id": selected_unit,
                    "dataset_type": dataset_type,
                    "features": features_list,
                    "method": "shap",
                    "top_k": 10
                },
                timeout=8.0
            )
            if r.status_code == 200:
                explain_res = r.json()
            else:
                st.error(f"API returned error: {r.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")

    if explain_res:
        contribs = explain_res.get("top_contributing_sensors", [])
        summary = explain_res.get("explanation_summary", "")

        # Plain-text Decision Brief
        st.markdown("#### 📝 Diagnostic Decision Brief")
        st.success(f"**EXPLAINABILITY SUMMARY:** {summary}")

        col_chart, col_table = st.columns([1.2, 1])

        with col_chart:
            st.markdown("#### 📊 TreeSHAP Attribution Magnitude")
            fig_shap = plot_shap_attributions(contribs)
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_table:
            st.markdown("#### 📑 Top Ranked Feature Contributions")
            if contribs:
                df_c = pd.DataFrame(contribs)
                # Format columns
                df_c["Impact Direction"] = df_c["contribution"].apply(
                    lambda x: "Increasing Degradation / Risk" if x == "positive" else "Protective Baseline"
                )
                st.dataframe(
                    df_c[["feature", "importance", "Impact Direction"]],
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("---")
        st.markdown("#### 🔬 Domain Grounding & Subsystem Correlation")
        if dataset_type == "NASA":
            st.markdown(
                """
                - **High-Pressure Compressor Outlet (`sensor_11` / T30):** Positive attribution indicates elevated core exit temperature, signaling blade erosion and thermodynamic efficiency decline.
                - **Low-Pressure Turbine Exhaust (`sensor_4`):** Increased exhaust temperature indicates reduced turbine expansion efficiency.
                - **Coolant Bleed Line (`sensor_13`):** Bleed valve pressure variations reflect bypass airflow leakage.
                """
            )
        else:
            st.markdown(
                """
                - **Mechanical Power:** Disproportionate power draw at constant RPM indicates spindle bearing binding or cutting insert bluntness.
                - **Tool Wear:** Cutting tip erosion exceeds nominal micro-meter tolerances, elevating catastrophic fracture probability.
                - **Torque to Speed Ratio:** Abnormal torque ratio spikes signify axis overstrain during feed transitions.
                """
            )
