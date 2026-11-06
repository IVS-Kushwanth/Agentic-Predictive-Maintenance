"""
Main Streamlit Application Entry Point.
Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework
for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support.
"""

import os
import sys
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.frontend.components import render_header, render_sidebar, validate_and_parse_csv, get_default_api_url
from app.frontend.pages import (
    render_overview,
    render_monitoring,
    render_predictive,
    render_explainability,
    render_agentic,
    render_recommendations,
    render_history_reports
)

# Page configuration
st.set_page_config(
    page_title="Agentic Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Render Sidebar and extract context
    selected_page, dataset_type, selected_unit, input_mode, uploaded_file = render_sidebar()
    api_url = st.session_state.get("api_url", get_default_api_url())

    # Render Header
    render_header(current_page=selected_page)

    # Handle custom CSV upload if active
    if input_mode == "Custom CSV Upload" and uploaded_file is not None:
        try:
            custom_data, feature_cols = validate_and_parse_csv(uploaded_file, dataset_type)
            st.success(
                f"✅ Validated custom CSV: Loaded {custom_data.shape[0]} rows with {custom_data.shape[1]} verified feature columns."
            )
            # Store in session state for active inspection
            st.session_state["custom_data"] = custom_data
            st.session_state["custom_features"] = feature_cols
        except Exception as e:
            st.error(f"❌ Schema Validation Error: {str(e)}")
            st.warning("Please upload a CSV file strictly matching the required training feature names.")

    # Page Dispatcher
    if "Overview" in selected_page:
        render_overview(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "Machine Monitoring" in selected_page:
        render_monitoring(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "Predictive Analysis" in selected_page:
        render_predictive(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "Explainable AI" in selected_page:
        render_explainability(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "Agentic AI Analysis" in selected_page:
        render_agentic(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "Maintenance Recommendations" in selected_page:
        render_recommendations(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)
    elif "History and Reports" in selected_page:
        render_history_reports(dataset_type=dataset_type, selected_unit=selected_unit, api_url=api_url)


if __name__ == "__main__":
    main()
