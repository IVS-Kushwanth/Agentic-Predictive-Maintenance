"""
Pages package for Streamlit Frontend.
"""

from .page_overview import render_overview
from .page_monitoring import render_monitoring
from .page_predictive import render_predictive
from .page_explainability import render_explainability
from .page_agentic import render_agentic
from .page_recommendations import render_recommendations
from .page_history import render_history_reports

__all__ = [
    "render_overview",
    "render_monitoring",
    "render_predictive",
    "render_explainability",
    "render_agentic",
    "render_recommendations",
    "render_history_reports"
]
