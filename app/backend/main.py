"""
FastAPI Main Application Entry Point.
Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework
for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support.
"""

import os
import sys
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.backend.routes import router

app = FastAPI(
    title="Agentic Predictive Maintenance API",
    description=(
        "Enterprise REST API for industrial predictive maintenance, Remaining Useful Life (RUL) estimation, "
        "machine failure risk analysis, multivariate sensor anomaly detection, Explainable AI (SHAP), "
        "and LangGraph multi-agent decision support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS safely from environment variables
raw_cors = os.environ.get("CORS_ORIGINS", "").strip()
if raw_cors:
    allowed_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]
    allow_credentials = "*" not in allowed_origins
else:
    # Default local development origins
    allowed_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=port, reload=False)
