"""
Recommendation History and Human Review Management using SQLite.
Implements persistence for generated maintenance recommendations, historical tracking,
and the human-in-the-loop review workflow (Approve, Reject, Request Further Analysis).
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .database import get_db_connection, init_db, DEFAULT_DB_PATH


class RecommendationHistoryDB:
    """
    Manages persistent SQLite storage of recommendation outcomes and human review cycles.
    """

    VALID_REVIEW_STATUSES = [
        "PENDING",
        "APPROVED",
        "REJECTED",
        "FURTHER_ANALYSIS_REQUESTED"
    ]

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        init_db(self.db_path)

    def save_recommendation(self, rec: Dict[str, Any]) -> str:
        """
        Persists a generated recommendation into the database with initial PENDING review status.
        Returns the unique recommendation_id.
        """
        rec_id = rec.get("recommendation_id") or f"REC-{uuid.uuid4().hex[:8].upper()}"
        machine_id = str(rec.get("machine_id", "UNKNOWN"))
        dataset_type = str(rec.get("dataset") or rec.get("dataset_type", "UNKNOWN"))
        prediction_date = rec.get("generated_at") or datetime.now(timezone.utc).isoformat()
        predicted_rul = rec.get("predicted_rul")
        failure_risk_score = rec.get("failure_risk_score")
        priority = str(rec.get("priority", "Informational"))
        risk_status = str(rec.get("risk_status", "nominal"))

        # Extract anomaly severity from evidence or metadata if available
        anomaly_severity = "Normal"
        for ev in rec.get("supporting_evidence", []):
            if ev.get("evidence_type") == "anomaly_score":
                anomaly_severity = ev.get("details", {}).get("severity", "Normal")
                break
        if "anomaly_severity" in rec:
            anomaly_severity = rec["anomaly_severity"]

        rec_copy = dict(rec)
        rec_copy["recommendation_id"] = rec_id
        recommendation_json = json.dumps(rec_copy, indent=2)
        created_at = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO recommendation_history (
                recommendation_id,
                machine_id,
                dataset_type,
                prediction_date,
                predicted_rul,
                failure_risk_score,
                priority,
                risk_status,
                anomaly_severity,
                recommendation_json,
                review_status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
        """, (
            rec_id,
            machine_id,
            dataset_type,
            prediction_date,
            predicted_rul,
            failure_risk_score,
            priority,
            risk_status,
            anomaly_severity,
            recommendation_json,
            created_at
        ))

        conn.commit()
        conn.close()
        return rec_id

    def get_recommendation(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single recommendation record by recommendation_id.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM recommendation_history
            WHERE recommendation_id = ?
        """, (recommendation_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        result = dict(row)
        try:
            result["recommendation_data"] = json.loads(result["recommendation_json"])
        except Exception:
            result["recommendation_data"] = {}

        return result

    def list_history(
        self,
        machine_id: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lists recommendation history, optionally filtered by machine_id and/or review_status.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM recommendation_history WHERE 1=1"
        params: List[Any] = []

        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)

        if review_status:
            query += " AND review_status = ?"
            params.append(review_status.upper())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            r_dict = dict(row)
            try:
                r_dict["recommendation_data"] = json.loads(r_dict["recommendation_json"])
            except Exception:
                r_dict["recommendation_data"] = {}
            results.append(r_dict)

        return results

    def update_review_status(
        self,
        recommendation_id: str,
        status: str,
        reviewed_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Updates the review status for a specific recommendation.
        Valid statuses: 'PENDING', 'APPROVED', 'REJECTED', 'FURTHER_ANALYSIS_REQUESTED'.
        """
        norm_status = status.upper().strip()
        if norm_status not in self.VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid status '{norm_status}'. Must be one of {self.VALID_REVIEW_STATUSES}"
            )

        if not reviewed_by or not reviewed_by.strip():
            raise ValueError("reviewed_by must be specified for human review tracking.")

        reviewed_at = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE recommendation_history
            SET review_status = ?,
                reviewed_by = ?,
                reviewed_at = ?,
                review_notes = ?
            WHERE recommendation_id = ?
        """, (norm_status, reviewed_by.strip(), reviewed_at, notes or "", recommendation_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def approve_recommendation(
        self,
        recommendation_id: str,
        reviewed_by: str,
        notes: str = ""
    ) -> bool:
        """Helper method to approve a maintenance recommendation."""
        return self.update_review_status(
            recommendation_id=recommendation_id,
            status="APPROVED",
            reviewed_by=reviewed_by,
            notes=notes
        )

    def reject_recommendation(
        self,
        recommendation_id: str,
        reviewed_by: str,
        notes: str = ""
    ) -> bool:
        """Helper method to reject a maintenance recommendation."""
        return self.update_review_status(
            recommendation_id=recommendation_id,
            status="REJECTED",
            reviewed_by=reviewed_by,
            notes=notes
        )

    def request_further_analysis(
        self,
        recommendation_id: str,
        reviewed_by: str,
        notes: str = ""
    ) -> bool:
        """Helper method to request further diagnostic testing or model analysis."""
        return self.update_review_status(
            recommendation_id=recommendation_id,
            status="FURTHER_ANALYSIS_REQUESTED",
            reviewed_by=reviewed_by,
            notes=notes
        )
