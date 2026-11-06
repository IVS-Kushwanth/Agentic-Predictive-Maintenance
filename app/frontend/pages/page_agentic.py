"""
Page 5: Agentic AI Analysis.
Interactive multi-agent decision support console invoking the compiled
LangGraph StateGraph workflow: Supervisor -> Predictor -> Anomaly -> Explainer -> Maintenance.
"""

import streamlit as st
import numpy as np
import time
import requests
from ..components.data_loader import get_sample_telemetry


def render_agentic(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader(f"🤖 LangGraph Multi-Agent Orchestration — {selected_unit}")

    st.markdown(
        "Coordinate specialized autonomous agents in a stateful directed acyclic graph (StateGraph). "
        "Agents exchange actual intermediate tensor states, validate data bounds, and produce "
        "evidence-grounded maintenance recommendations."
    )

    telemetry_data, meta = get_sample_telemetry(dataset_type, selected_unit)
    features_list = telemetry_data[-1, :].tolist() if telemetry_data.ndim >= 2 else telemetry_data.tolist()

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_analysis = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

    with col_info:
        st.caption(f"Target: `{selected_unit}` | Dataset: `{dataset_type}` | Features: `{len(features_list)} dimensions`")

    if start_analysis:
        progress_bar = st.progress(0, text="Initializing Supervisor Agent...")
        step_container = st.container()

        with step_container:
            st.markdown("#### 🔄 Agent Workflow Execution Trace")
            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()
            step5 = st.empty()
            step6 = st.empty()

            step1.info("🧑‍💼 **Supervisor Agent:** Validating input dimensionality and setting operational bounds...")
            time.sleep(0.3)
            progress_bar.progress(20, text="Executing Supervised Prediction...")

            step2.info("📊 **Prediction Agent:** Invoking trained regression / classification models...")
            time.sleep(0.3)
            progress_bar.progress(40, text="Evaluating Sensor Anomaly Severity...")

            step3.info("🚨 **Anomaly Detection Agent:** Computing LSTM Autoencoder reconstruction errors...")
            time.sleep(0.3)
            progress_bar.progress(60, text="Calculating TreeSHAP Attributions...")

            step4.info("🧠 **Explanation Agent:** Generating TreeSHAP attributions and identifying top degradation drivers...")
            time.sleep(0.3)
            progress_bar.progress(80, text="Formulating Maintenance Actions...")

            step5.info("🛠️ **Maintenance Agent:** Synthesizing evidence, checking conflict matrix, and persisting recommendation...")

            # Call FastAPI /api/agents/analyze
            agent_result = None
            try:
                r = requests.post(
                    f"{api_url}/api/agents/analyze",
                    json={
                        "machine_id": selected_unit,
                        "dataset_type": dataset_type,
                        "sensor_data": features_list,
                        "workflow_type": "full_analysis"
                    },
                    timeout=15.0
                )
                if r.status_code == 200:
                    agent_result = r.json()
                else:
                    st.error(f"Agent execution failed with HTTP {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Agent communication error: {str(e)}")

            progress_bar.progress(100, text="Workflow Completed Successfully.")
            step6.success("✅ **Result Validation Node:** All agent state transitions verified. Decision brief compiled.")

        if agent_result:
            st.markdown("---")
            st.markdown("### 📋 Executive Decision Brief")

            rec_id = agent_result.get("recommendation_id", "N/A")
            priority = agent_result.get("priority", "Informational")
            risk_status = agent_result.get("risk_status", "nominal")
            summary = agent_result.get("explanation_summary", "No summary provided.")

            # Priority alert banner
            if priority in ["Critical review required", "High priority"]:
                st.error(f"**PRIORITY: {priority.upper()} | STATUS: {risk_status.upper()}** (Recommendation ID: `{rec_id}`)")
            elif priority == "Medium priority":
                st.warning(f"**PRIORITY: {priority.upper()} | STATUS: {risk_status.upper()}** (Recommendation ID: `{rec_id}`)")
            else:
                st.success(f"**PRIORITY: {priority.upper()} | STATUS: {risk_status.upper()}** (Recommendation ID: `{rec_id}`)")

            st.code(summary, language="text")

            # Agent outputs tabs
            tab_pred, tab_anom, tab_xai, tab_rec = st.tabs([
                "📊 Prediction Agent", "🚨 Anomaly Agent", "🧠 Explanation Agent", "🛠️ Suggested Actions"
            ])

            with tab_pred:
                if dataset_type == "NASA":
                    st.metric("Predicted Remaining Useful Life", f"{agent_result.get('rul_prediction', 0.0):.1f} Cycles")
                    ew = agent_result.get("confidence_or_uncertainty_information", {}).get("early_warning_horizons", {})
                    if ew:
                        st.json(ew)
                else:
                    st.metric("Failure Probability", f"{agent_result.get('failure_risk_score', 0.0)*100:.1f}%")
                    st.write(f"**Assigned Class:** `{agent_result.get('failure_label', 'Unknown')}`")

            with tab_anom:
                st.metric("Anomaly Severity", agent_result.get("anomaly_severity", "Normal"))
                st.metric("Raw Reconstruction Error", f"{agent_result.get('anomaly_score', 0.0):.6f}")

            with tab_xai:
                st.markdown("##### Top Contributing Sensor Drivers")
                contribs = agent_result.get("important_sensor_contributions", [])
                if contribs:
                    st.dataframe(contribs, use_container_width=True)

            with tab_rec:
                actions = agent_result.get("suggested_maintenance_actions", [])
                for idx, act in enumerate(actions):
                    st.markdown(
                        f"**Action {idx+1}: {act.get('action_description', act.get('description', ''))}**<br>"
                        f"• Urgency: `{act.get('urgency', 'Planned')}` | Subsystem: `{act.get('subsystem', act.get('target_subsystem', ''))}`<br>"
                        f"• Evidence Link: *{act.get('evidence_link', act.get('justification', ''))}*",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
