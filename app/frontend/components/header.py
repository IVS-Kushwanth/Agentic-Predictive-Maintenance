"""
Header banner and custom CSS for Streamlit Frontend.
Provides consistent industrial styling, badge indicators, and clean typography.
"""

import streamlit as st


def render_header(current_page: str = "Overview"):
    """Renders the top branding header with operational status."""
    st.markdown("""
        <style>
        /* Modern Industrial Typography & Color Theme */
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border-left: 6px solid #3B82F6;
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            color: #F8FAFC;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.025em;
        }
        .main-header p {
            color: #94A3B8;
            font-size: 0.90rem;
            margin: 0;
        }
        .status-badge {
            display: inline-block;
            background-color: #064E3B;
            color: #34D399;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid #059669;
        }
        .breadcrumb {
            font-size: 0.80rem;
            color: #64748B;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        /* Custom Metric Cards */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .metric-title {
            color: #94A3B8;
            font-size: 0.80rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0.25rem 0;
        }
        .metric-sub {
            color: #64748B;
            font-size: 0.75rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="breadcrumb">Fleet Diagnostics / {current_page}</div>
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>Agentic Predictive Maintenance Platform</h1>
                    <p>Explainable Time-Series Anomaly Detection, Remaining Useful Life Estimation & Multi-Agent Decision Support</p>
                </div>
                <div>
                    <span class="status-badge">System Operational</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
