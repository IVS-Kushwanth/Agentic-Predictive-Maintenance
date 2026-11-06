"""
Database connector and session manager for Phase 8 Web Application.
Reuses the centralized SQLite database at data/maintenance_history.db.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database.database import get_db_connection, init_db, DEFAULT_DB_PATH
from src.database.recommendation_history import RecommendationHistoryDB

__all__ = ["get_db_connection", "init_db", "DEFAULT_DB_PATH", "RecommendationHistoryDB"]
