"""
Pydantic Request and Response Schemas for Phase 8 FastAPI Backend.
Enforces strict schema validation, type safety, and helpful error messages.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Common / Health Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Agentic Predictive Maintenance API"
    version: str = "1.0.0"
    timestamp: str


class ModelMetadata(BaseModel):
    model_id: str
    dataset: Literal["NASA", "AI4I"]
    task: Literal["rul_regression", "failure_classification", "anomaly_detection", "early_warning"]
    framework: str
    description: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ModelsListResponse(BaseModel):
    total_models: int
    models: List[ModelMetadata]


# ---------------------------------------------------------------------------
# Prediction Schemas
# ---------------------------------------------------------------------------

class NASARULPredictRequest(BaseModel):
    machine_id: str = Field(..., description="Engine identifier, e.g. 'ENG-101'")
    sensor_features: List[float] = Field(..., description="Feature vector with 153 tabular features or 30x14 temporal window")
    model_type: Optional[str] = Field("xgboost", description="'xgboost', 'random_forest', 'gradient_boosting', or 'lstm'")


class NASARULPredictResponse(BaseModel):
    machine_id: str
    dataset: str = "NASA"
    predicted_rul: float
    unit: str = "cycles"
    model_used: str
    early_warning_status: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.85


class AI4IFailurePredictRequest(BaseModel):
    machine_id: str = Field(..., description="Machine tool identifier, e.g. 'CNC-202'")
    operating_features: List[float] = Field(..., description="Engineered feature vector with exactly 14 features")
    model_type: Optional[str] = Field("xgboost", description="'xgboost', 'random_forest', or 'gradient_boosting'")


class AI4IFailurePredictResponse(BaseModel):
    machine_id: str
    dataset: str = "AI4I"
    failure_risk_score: float = Field(..., description="Calibrated breakdown probability between 0.0 and 1.0")
    failure_prediction: int = Field(..., description="0 for Normal, 1 for Failure")
    failure_label: str
    model_used: str
    is_calibrated: bool = True
    confidence: float = 0.88


# ---------------------------------------------------------------------------
# Anomaly Detection Schemas
# ---------------------------------------------------------------------------

class AnomalyDetectRequest(BaseModel):
    machine_id: str
    dataset_type: Literal["NASA", "AI4I"]
    features: List[float] = Field(..., description="Sensor or operating condition values")
    detector_type: Optional[str] = Field("autoencoder", description="'autoencoder' or 'isolation_forest'")


class AnomalyDetectResponse(BaseModel):
    machine_id: str
    dataset: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    severity: Literal["Normal", "Low anomaly", "Medium anomaly", "High anomaly"]
    detector_used: str


# ---------------------------------------------------------------------------
# Explainability Schemas
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    machine_id: str
    dataset_type: Literal["NASA", "AI4I"]
    features: List[float]
    method: Optional[str] = Field("shap", description="'shap' or 'permutation'")
    top_k: Optional[int] = Field(5, description="Number of top features to return")


class SensorAttribution(BaseModel):
    feature: str
    importance: float
    contribution: Literal["positive", "negative"]
    value: Optional[float] = None


class ExplainResponse(BaseModel):
    machine_id: str
    dataset: str
    method: str
    top_contributing_sensors: List[SensorAttribution]
    explanation_summary: str


# ---------------------------------------------------------------------------
# Multi-Agent LangGraph Schemas
# ---------------------------------------------------------------------------

class AgentAnalyzeRequest(BaseModel):
    machine_id: str
    dataset_type: Literal["NASA", "AI4I"]
    sensor_data: List[float] = Field(..., description="Raw sensor vector (153 features for NASA, 14 for AI4I)")
    workflow_type: Optional[str] = Field("full_analysis", description="'full_analysis', 'risk_only', 'rul_only'")


class AgentAnalyzeResponse(BaseModel):
    request_id: Optional[str] = None
    dataset: str
    machine_id: str
    workflow_status: str
    recommendation_id: Optional[str] = None
    priority: Optional[str] = None
    risk_status: Optional[str] = None
    rul_prediction: Optional[float] = None
    failure_risk_score: Optional[float] = None
    failure_label: Optional[str] = None
    anomaly_status: str
    anomaly_severity: str
    anomaly_score: Optional[float] = None
    important_sensor_contributions: List[Dict[str, Any]] = Field(default_factory=list)
    explanation_summary: Optional[str] = None
    suggested_maintenance_actions: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_or_uncertainty_information: Dict[str, Any] = Field(default_factory=dict)
    evidence_references: Dict[str, Any] = Field(default_factory=dict)
    human_review_status: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Recommendation & Review Schemas
# ---------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    dataset_type: Literal["NASA", "AI4I"]
    machine_id: str
    predicted_rul: Optional[float] = None
    failure_risk_score: Optional[float] = None
    anomaly_severity: str = "Normal"
    anomaly_score: Optional[float] = None
    important_sensors: Optional[List[Dict[str, Any]]] = None
    model_confidence: Optional[float] = None


class RecommendationResponse(BaseModel):
    recommendation_id: Optional[str] = None
    machine_id: str
    dataset: str
    priority: str
    predicted_rul: Optional[float] = None
    failure_risk_score: Optional[float] = None
    confidence_score: Optional[float] = None
    risk_status: str
    reason: List[str]
    recommended_actions: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    human_approval_required: bool
    disclaimer: str
    generated_at: str


class ReviewRequest(BaseModel):
    recommendation_id: str
    status: Literal["APPROVED", "REJECTED", "FURTHER_ANALYSIS_REQUESTED"]
    reviewed_by: str = Field(..., min_length=2, description="Certified engineer name/ID")
    review_notes: Optional[str] = Field("", description="Engineering justification or operational notes")


class ReviewResponse(BaseModel):
    recommendation_id: str
    review_status: str
    reviewed_by: str
    reviewed_at: str
    review_notes: str
    success: bool


class HistoryItem(BaseModel):
    id: int
    recommendation_id: str
    machine_id: str
    dataset_type: str
    prediction_date: str
    predicted_rul: Optional[float] = None
    failure_risk_score: Optional[float] = None
    priority: str
    risk_status: str
    anomaly_severity: str
    review_status: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: str
    recommendation_data: Dict[str, Any] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    total_records: int
    records: List[HistoryItem]
