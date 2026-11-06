"""
FastAPI Routes for Phase 8 Predictive Maintenance API.
Implements endpoints:
- GET /health
- GET /api/models
- POST /api/predict/rul
- POST /api/predict/failure
- POST /api/anomaly/detect
- POST /api/explain
- POST /api/agents/analyze
- POST /api/recommendations
- GET /api/history
- POST /api/recommendations/review
"""

from typing import Optional, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, status

from app.backend.schemas import (
    HealthResponse,
    ModelsListResponse,
    NASARULPredictRequest,
    NASARULPredictResponse,
    AI4IFailurePredictRequest,
    AI4IFailurePredictResponse,
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    ExplainRequest,
    ExplainResponse,
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    RecommendationRequest,
    RecommendationResponse,
    ReviewRequest,
    ReviewResponse,
    HistoryResponse,
    HistoryItem
)

from app.backend.services import (
    get_all_models_metadata,
    PredictionService,
    AnomalyService,
    ExplainService,
    AgentService,
    HistoryService
)
from src.recommendations import RecommendationEngine, RecommendationInput

router = APIRouter()
_pred_service = PredictionService()
_anom_service = AnomalyService()
_explain_service = ExplainService()
_agent_service = AgentService()
_history_service = HistoryService()
_rec_engine = RecommendationEngine()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint confirming API service readiness."""
    return {
        "status": "healthy",
        "service": "Agentic Predictive Maintenance API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/api/models", response_model=ModelsListResponse, tags=["Models"])
def list_models():
    """Returns catalog of all trained baseline, deep learning, and anomaly detection models."""
    models = get_all_models_metadata()
    return {
        "total_models": len(models),
        "models": models
    }


@router.post("/api/predict/rul", response_model=NASARULPredictResponse, tags=["Prediction"])
def predict_nasa_rul(request: NASARULPredictRequest):
    """Predicts Remaining Useful Life (RUL) for NASA C-MAPSS turbofan engines."""
    try:
        res = _pred_service.predict_nasa_rul(
            machine_id=request.machine_id,
            sensor_features=request.sensor_features,
            model_type=request.model_type
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"NASA prediction failed: {str(e)}")


@router.post("/api/predict/failure", response_model=AI4IFailurePredictResponse, tags=["Prediction"])
def predict_ai4i_failure(request: AI4IFailurePredictRequest):
    """Predicts machine breakdown probability for UCI AI4I 2020 CNC machines."""
    try:
        res = _pred_service.predict_ai4i_failure(
            machine_id=request.machine_id,
            operating_features=request.operating_features,
            model_type=request.model_type
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI4I prediction failed: {str(e)}")


@router.post("/api/anomaly/detect", response_model=AnomalyDetectResponse, tags=["Anomaly Detection"])
def detect_anomaly(request: AnomalyDetectRequest):
    """Runs unsupervised anomaly detection using autoencoders or isolation forests."""
    try:
        res = _anom_service.detect_anomaly(
            machine_id=request.machine_id,
            dataset_type=request.dataset_type,
            features=request.features,
            detector_type=request.detector_type
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/api/explain", response_model=ExplainResponse, tags=["Explainable AI"])
def explain_prediction(request: ExplainRequest):
    """Computes TreeSHAP feature attributions and sensor contributions."""
    try:
        res = _explain_service.explain(
            machine_id=request.machine_id,
            dataset_type=request.dataset_type,
            features=request.features,
            method=request.method,
            top_k=request.top_k or 5
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Explainability computation failed: {str(e)}")


@router.post("/api/agents/analyze", response_model=AgentAnalyzeResponse, tags=["Agentic AI"])
def run_agentic_analysis(request: AgentAnalyzeRequest):
    """Executes full multi-agent LangGraph orchestration workflow."""
    try:
        res = _agent_service.analyze(
            machine_id=request.machine_id,
            dataset_type=request.dataset_type,
            sensor_data=request.sensor_data,
            workflow_type=request.workflow_type or "full_analysis"
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent workflow execution failed: {str(e)}")


@router.post("/api/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
def generate_recommendation(request: RecommendationRequest):
    """Generates evidence-linked maintenance suggestions and persists record in SQLite."""
    try:
        payload = RecommendationInput(
            dataset_type=request.dataset_type,
            machine_id=request.machine_id,
            predicted_rul=request.predicted_rul,
            failure_risk_score=request.failure_risk_score,
            anomaly_severity=request.anomaly_severity,
            anomaly_score=request.anomaly_score,
            important_sensors=request.important_sensors,
            model_confidence=request.model_confidence
        )
        rec = _rec_engine.generate_recommendation(payload)
        rec_id = _history_service.db.save_recommendation(rec)
        rec["recommendation_id"] = rec_id
        return rec
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Recommendation generation failed: {str(e)}")


@router.get("/api/history", response_model=HistoryResponse, tags=["History & Audit"])
def get_recommendation_history(
    machine_id: Optional[str] = Query(None, description="Filter by machine/engine identifier"),
    review_status: Optional[str] = Query(None, description="Filter by review status: PENDING, APPROVED, REJECTED, FURTHER_ANALYSIS_REQUESTED"),
    limit: int = Query(50, ge=1, le=200, description="Max records to retrieve")
):
    """Retrieves previous analysis records, predictions, and human review history."""
    try:
        records = _history_service.get_history(machine_id=machine_id, review_status=review_status, limit=limit)
        return {
            "total_records": len(records),
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query history: {str(e)}")


@router.post("/api/recommendations/review", response_model=ReviewResponse, tags=["Human Review"])
def review_recommendation(request: ReviewRequest):
    """Records certified human engineer authorization (Approve, Reject, or Request Further Analysis)."""
    try:
        res = _history_service.review(
            recommendation_id=request.recommendation_id,
            status=request.status,
            reviewed_by=request.reviewed_by,
            notes=request.review_notes
        )
        if not res["success"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recommendation ID '{request.recommendation_id}' not found.")
        return res
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Review update failed: {str(e)}")
