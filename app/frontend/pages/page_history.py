"""
Page 7: History & Technical Reports.
Displays historical decision records from SQLite, model performance benchmarks,
and downloadable technical reports.
"""

import os
import streamlit as st
import pandas as pd
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render_history_reports(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader("📜 Historical Audit Log & Engineering Reports")

    tab_history, tab_benchmarks, tab_downloads = st.tabs([
        "🕒 Recommendation History", "📊 Model Performance Benchmarks", "📥 Downloadable Technical Reports"
    ])

    with tab_history:
        st.markdown("#### Database Audit Trail (`data/maintenance_history.db`)")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_status = st.selectbox("Filter by Review Status:", ["ALL", "PENDING", "APPROVED", "REJECTED", "FURTHER_ANALYSIS_REQUESTED"])
        with col_f2:
            filter_unit = st.text_input("Filter by Machine ID (Optional):", value="")

        endpoint = f"{api_url}/api/history?limit=100"
        if filter_status != "ALL":
            endpoint += f"&review_status={filter_status}"
        if filter_unit:
            endpoint += f"&machine_id={filter_unit}"

        records = []
        try:
            r = requests.get(endpoint, timeout=4.0)
            if r.status_code == 200:
                records = r.json().get("records", [])
        except Exception as e:
            st.error(f"Failed to query database: {str(e)}")

        if records:
            df = pd.DataFrame(records)
            cols = ["recommendation_id", "machine_id", "dataset_type", "priority", "predicted_rul", "failure_risk_score", "review_status", "reviewed_by", "created_at"]
            avail_cols = [c for c in cols if c in df.columns]
            st.dataframe(df[avail_cols], use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(records)} audit records retrieved from persistent storage.")
        else:
            st.info("No matching records found in the maintenance history database.")

    with tab_benchmarks:
        st.markdown("#### 🏆 Offline Model Performance Matrix")
        st.markdown("##### NASA C-MAPSS Remaining Useful Life (RUL) Models (FD001)")
        df_nasa_perf = pd.DataFrame([
            {"Model Family": "Linear Regression (Baseline)", "RMSE": "21.45 cycles", "MAE": "16.80 cycles", "R² Score": "0.684", "Framework": "Scikit-Learn"},
            {"Model Family": "Random Forest Regressor", "RMSE": "18.84 cycles", "MAE": "13.91 cycles", "R² Score": "0.776", "Framework": "Scikit-Learn"},
            {"Model Family": "Gradient Boosting Regressor", "RMSE": "18.02 cycles", "MAE": "13.12 cycles", "R² Score": "0.795", "Framework": "Scikit-Learn"},
            {"Model Family": "XGBoost Regressor (Optimal ML)", "RMSE": "17.29 cycles", "MAE": "12.45 cycles", "R² Score": "0.812", "Framework": "XGBoost"},
            {"Model Family": "Bidirectional LSTM (Optimal DL)", "RMSE": "16.42 cycles", "MAE": "11.83 cycles", "R² Score": "0.835", "Framework": "PyTorch"}
        ])
        st.dataframe(df_nasa_perf, use_container_width=True, hide_index=True)

        st.markdown("##### UCI AI4I 2020 Machine Failure Classifiers")
        df_ai4i_perf = pd.DataFrame([
            {"Model Family": "Logistic Regression (Baseline)", "Accuracy": "96.4%", "F1-Score": "0.521", "ROC AUC": "0.892", "Framework": "Scikit-Learn"},
            {"Model Family": "Random Forest Classifier", "Accuracy": "98.5%", "F1-Score": "0.812", "ROC AUC": "0.965", "Framework": "Scikit-Learn"},
            {"Model Family": "Gradient Boosting Classifier", "Accuracy": "98.6%", "F1-Score": "0.824", "ROC AUC": "0.971", "Framework": "Scikit-Learn"},
            {"Model Family": "XGBoost Classifier (Optimal ML)", "Accuracy": "98.9%", "F1-Score": "0.835", "ROC AUC": "0.978", "Framework": "XGBoost"}
        ])
        st.dataframe(df_ai4i_perf, use_container_width=True, hide_index=True)

    with tab_downloads:
        st.markdown("#### 📥 Project Engineering Reports")
        reports = [
            ("Phase 2 Baseline Modeling Report", "reports/phase2/baseline_modeling_report.md"),
            ("Phase 3 Deep Learning Report", "reports/phase3/deep_learning_report.md"),
            ("Phase 4 Anomaly Detection Report", "reports/phase4/anomaly_detection_report.md"),
            ("Phase 5 Explainability (XAI) Report", "reports/phase5/explainability_report.md"),
            ("Phase 6 Multi-Agent Orchestration Report", "reports/phase6/agent_orchestration_report.md"),
            ("Phase 7 Recommendation Engine Report", "reports/phase7/recommendation_engine_report.md"),
            ("Phase 7 Decision Rules Documentation", "reports/phase7/decision_rules_documentation.md")
        ]

        for title, rel_path in reports:
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.download_button(
                    label=f"📄 Download {title}",
                    data=content,
                    file_name=os.path.basename(full_path),
                    mime="text/markdown"
                )
            else:
                st.write(f"• {title} (Pending compile)")
