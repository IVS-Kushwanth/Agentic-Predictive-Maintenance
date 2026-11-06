"""
SQLite database connection and schema initialization for Recommendation History and Review Tracking.
"""

import os
import sqlite3
from typing import Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "maintenance_history.db")


def resolve_db_path(db_path: Optional[str] = None) -> str:
    """Resolves the database path from argument, DATABASE_PATH env var, or local default."""
    if db_path:
        return db_path
    env_path = os.environ.get("DATABASE_PATH")
    if env_path and env_path.strip():
        # Handle relative or absolute paths safely
        return os.path.abspath(env_path.strip()) if not os.path.isabs(env_path.strip()) else env_path.strip()
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Returns an ACID-compliant SQLite connection with row_factory set to Row."""
    path = resolve_db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        conn = sqlite3.connect(path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise ConnectionError(f"Failed to connect to SQLite database at {path}: {str(e)}") from e


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema if not already present."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recommendation_id TEXT UNIQUE NOT NULL,
        machine_id TEXT NOT NULL,
        dataset_type TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        predicted_rul REAL,
        failure_risk_score REAL,
        priority TEXT NOT NULL,
        risk_status TEXT NOT NULL,
        anomaly_severity TEXT NOT NULL,
        recommendation_json TEXT NOT NULL,
        review_status TEXT NOT NULL DEFAULT 'PENDING',
        reviewed_by TEXT,
        reviewed_at TEXT,
        review_notes TEXT,
        created_at TEXT NOT NULL
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_machine_id ON recommendation_history(machine_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON recommendation_history(review_status);")

    conn.commit()
    conn.close()
