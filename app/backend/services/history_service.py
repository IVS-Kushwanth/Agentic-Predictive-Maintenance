"""
History and Review Service for Phase 8 FastAPI Backend.
Handles querying historical recommendations and processing certified human engineer
reviews (approval, rejection, further analysis requests).
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .model_cache import get_history_db


class HistoryService:
    """Manages SQLite recommendation history and audit trails."""

    def __init__(self):
        self.db = get_history_db()

    def get_history(
        self,
        machine_id: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        return self.db.list_history(machine_id=machine_id, review_status=review_status, limit=limit)

    def review(
        self,
        recommendation_id: str,
        status: str,
        reviewed_by: str,
        notes: Optional[str] = ""
    ) -> Dict[str, Any]:
        success = self.db.update_review_status(
            recommendation_id=recommendation_id,
            status=status,
            reviewed_by=reviewed_by,
            notes=notes
        )

        return {
            "recommendation_id": recommendation_id,
            "review_status": status,
            "reviewed_by": reviewed_by,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_notes": notes or "",
            "success": success
        }
