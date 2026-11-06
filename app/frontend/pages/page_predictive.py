"""
Page 3: Predictive Analysis.
Displays real model inference outputs: NASA RUL regression, AI4I failure risk
with calibration disclosures, early-warning classification, and anomaly scores.
"""

import streamlit as st
import numpy as np
import requests
from ..components.data_loader import get_sample_telemetry
from ..components.charts import plot_gauge_meter, plot_early_warning_distribution


def render_predictive(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader(f"📈 Predictive Analytics & Condition Assessment — {selected_unit}")

    telemetry_data, meta = get_sample_telemetry(dataset_type, selected_unit)

    # Call FastAPI endpoints
    col1, col2 = st.columns([1, 1])

    if dataset_type == "NASA":
        # Extract 153 features from terminal cycle
        terminal_features = telemetry_data[-1, :].tolist() if telemetry_data.ndim >= 2 else telemetry_data.tolist()

        pred_res = None
        anom_res = None

        with st.spinner("Invoking NASA predictive baseline & anomaly models..."):
            try:
                r_pred = requests.post(
                    f"{api_url}/api/predict/rul",
                    json={"machine_id": selected_unit, "sensor_features": terminal_features},
                    timeout=5.0
                )
                if r_pred.status_code == 200:
                    pred_res = r_pred.json()

                r_anom = requests.post(
                    f"{api_url}/api/anomaly/detect",
                    json={"machine_id": selected_unit, "dataset_type": "NASA", "features": terminal_features},
                    timeout=5.0
                )
                if r_anom.status_code == 200:
                    anom_res = r_anom.json()
            except Exception as e:
                st.error(f"Backend API error: {str(e)}")

        if pred_res:
            rul = pred_res["predicted_rul"]
            ew = pred_res.get("early_warning_status", {})

            with col1:
                st.markdown("#### ⏳ Remaining Useful Life (RUL)")
                st.metric(
                    label="Predicted RUL",
                    value=f"{rul:.1f} Cycles",
                    delta=f"{rul - meta['actual_rul']:+.1f} Error vs Ground Truth ({meta['actual_rul']:.1f})",
                    delta_color="off"
                )
                st.markdown(
                    f"**Inference Model:** `{pred_res['model_used']}`<br>"
                    f"**Model Certainty Bound:** `{pred_res['confidence']*100:.0f}%`",
                    unsafe_allow_html=True
                )

                if ew:
                    fig_ew = plot_early_warning_distribution(ew)
                    st.plotly_chart(fig_ew, use_container_width=True)

            with col2:
                st.markdown("#### 🚨 Multivariate Anomaly Score")
                if anom_res:
                    score = anom_res["anomaly_score"]
                    th = anom_res["threshold"]
                    sev = anom_res["severity"]

                    st.metric(
                        label="Anomaly Severity",
                        value=sev,
                        delta=f"Reconstruction Error: {score:.5f} (Threshold: {th:.5f})",
                        delta_color="inverse" if anom_res["is_anomaly"] else "normal"
                    )

                    fig_gauge = plot_gauge_meter(
                        value=min(1.0, score / (th * 2.5)),
                        title="Sensor Drift Index",
                        max_val=1.0,
                        thresholds=[0.4, 0.7]
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.markdown(f"**Detector Architecture:** `{anom_res['detector_used']}`")

    else:
        # AI4I Flow
        features_list = telemetry_data.tolist()
        pred_res = None
        anom_res = None

        with st.spinner("Invoking AI4I breakdown classifier & anomaly models..."):
            try:
                r_pred = requests.post(
                    f"{api_url}/api/predict/failure",
                    json={"machine_id": selected_unit, "operating_features": features_list},
                    timeout=5.0
                )
                if r_pred.status_code == 200:
                    pred_res = r_pred.json()

                r_anom = requests.post(
                    f"{api_url}/api/anomaly/detect",
                    json={"machine_id": selected_unit, "dataset_type": "AI4I", "features": features_list},
                    timeout=5.0
                )
                if r_anom.status_code == 200:
                    anom_res = r_anom.json()
            except Exception as e:
                st.error(f"Backend API error: {str(e)}")

        if pred_res:
            risk = pred_res["failure_risk_score"]
            label = pred_res["failure_label"]

            with col1:
                st.markdown("#### 💥 Machine Breakdown Risk")
                st.metric(
                    label="Failure Probability",
                    value=f"{risk*100:.1f}%",
                    delta=label,
                    delta_color="inverse" if label == "Machine Failure" else "normal"
                )
                st.info(
                    "**CALIBRATION DISCLOSURE:** This probability score is **isotonic-calibrated** against "
                    "Phase 2 validation loss curves. Decision threshold calibrated at **0.7969**."
                )
                fig_gauge_risk = plot_gauge_meter(
                    value=risk,
                    title="Breakdown Probability Gauge",
                    max_val=1.0,
                    thresholds=[0.40, 0.75]
                )
                st.plotly_chart(fig_gauge_risk, use_container_width=True)

            with col2:
                st.markdown("#### 🔍 Operating Outlier Severity")
                if anom_res:
                    sev = anom_res["severity"]
                    st.metric(
                        label="Anomaly Status",
                        value=sev,
                        delta="Outlier Regime" if anom_res["is_anomaly"] else "Normal Operating Envelope",
                        delta_color="inverse" if anom_res["is_anomaly"] else "normal"
                    )
                    st.markdown(
                        f"**Raw Outlier Score:** `{anom_res['anomaly_score']:.5f}`<br>"
                        f"**Detector Used:** `{anom_res['detector_used']}`",
                        unsafe_allow_html=True
                    )
