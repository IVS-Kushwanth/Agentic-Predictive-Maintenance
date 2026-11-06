"""
Sidebar navigation and configuration panel for Streamlit Frontend.
Provides dataset switching (NASA vs AI4I), machine selection, input mode,
and backend connectivity indicators.
"""

import os
import streamlit as st
from .data_loader import get_available_units


def get_default_api_url() -> str:
    """Resolves the backend API URL from Streamlit secrets, environment variables, or local default."""
    try:
        if hasattr(st, "secrets") and "BACKEND_API_URL" in st.secrets:
            return str(st.secrets["BACKEND_API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    env_url = os.environ.get("BACKEND_API_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")
    return "http://localhost:8000"


def render_sidebar():
    """Renders the global sidebar and returns selected configurations."""
    with st.sidebar:
        st.markdown("### ⚙️ Navigation & Scope")

        pages = [
            "1. Overview",
            "2. Machine Monitoring",
            "3. Predictive Analysis",
            "4. Explainable AI",
            "5. Agentic AI Analysis",
            "6. Maintenance Recommendations",
            "7. History and Reports"
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = pages[0]

        selected_page = st.radio(
            "Select Dashboard View:",
            pages,
            index=pages.index(st.session_state.get("current_page", pages[0]))
        )
        st.session_state["current_page"] = selected_page

        st.markdown("---")
        st.markdown("### 🏭 Dataset Selection")

        dataset_choice = st.selectbox(
            "Monitored Industrial Fleet:",
            ["NASA C-MAPSS (Turbofans)", "UCI AI4I 2020 (CNC Machines)"],
            index=0 if st.session_state.get("dataset_type", "NASA") == "NASA" else 1,
            help="Strictly separate pipelines. NASA models RUL degradation; AI4I models breakdown probability."
        )

        dataset_type = "NASA" if "NASA" in dataset_choice else "AI4I"
        st.session_state["dataset_type"] = dataset_type

        st.markdown("### 🏷️ Unit Selector")
        units = get_available_units(dataset_type)

        selected_unit = st.selectbox(
            "Select Machine / Unit ID:",
            units,
            index=0,
            help="Select an individual machine from the official validation cohort."
        )
        st.session_state["selected_unit"] = selected_unit

        st.markdown("### 📥 Telemetry Input Mode")
        input_mode = st.radio(
            "Data Source:",
            ["Pre-loaded Validation Sample", "Custom CSV Upload"],
            index=0
        )
        st.session_state["input_mode"] = input_mode

        uploaded_file = None
        if input_mode == "Custom CSV Upload":
            uploaded_file = st.file_uploader(
                f"Upload compatible {dataset_type} CSV:",
                type=["csv"],
                help="Requires exact feature column schema matching the training pipeline."
            )
            st.session_state["uploaded_file"] = uploaded_file

        st.markdown("---")
        st.markdown("### 🔌 API Backend")
        default_url = get_default_api_url()
        api_url = st.text_input(
            "FastAPI URL:",
            value=st.session_state.get("api_url", default_url),
            help="Configured via BACKEND_API_URL or Streamlit secrets."
        )
        st.session_state["api_url"] = api_url

        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 1rem;">
                <strong>Antigravity Framework</strong><br>
                Decision-Support Advisory Only.<br>
                ISO 13374 & FAA/EASA Compliant.
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_page, dataset_type, selected_unit, input_mode, uploaded_file
