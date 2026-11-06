# ==============================================================================
# Agentic Predictive Maintenance Framework — Production Multi-Stage Dockerfile
# ==============================================================================
# Python 3.12 Slim Linux Base Image
# Optimized for CPU-only inference, non-root security, and modular services
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base Runtime & Dependency Installation
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal OS build tools and curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheel explicitly to prevent massive CUDA package bloat
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install application dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create an unprivileged non-root user (UID 1000) for security compliance
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy application code, configuration, models, and runtime validation datasets
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser run_dashboard.py .

# Copy ONLY runtime validation datasets (15.2 MB) — avoiding 1.84 GB training archives
COPY --chown=appuser:appuser data/processed/nasa/FD001/sequences_w30_val.npz data/processed/nasa/FD001/
COPY --chown=appuser:appuser data/processed/nasa/FD001/val_tabular.csv data/processed/nasa/FD001/
COPY --chown=appuser:appuser data/processed/nasa/FD001/metadata.json data/processed/nasa/FD001/
COPY --chown=appuser:appuser data/processed/ai4i/ data/processed/ai4i/

USER appuser

# ------------------------------------------------------------------------------
# Target: Backend (FastAPI Microservice)
# ------------------------------------------------------------------------------
FROM base AS backend

ENV PORT=8000 \
    DATABASE_PATH=/app/data/maintenance_history.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT}"]

# ------------------------------------------------------------------------------
# Target: Frontend (Streamlit Dashboard)
# ------------------------------------------------------------------------------
FROM base AS frontend

ENV STREAMLIT_PORT=8501 \
    BACKEND_API_URL=http://backend:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${STREAMLIT_PORT}/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run app/frontend/main.py --server.port ${STREAMLIT_PORT} --server.address 0.0.0.0 --server.headless true"]

# ------------------------------------------------------------------------------
# Target: All-In-One (Single-Container PaaS Deployments e.g. Render / Cloud Run)
# ------------------------------------------------------------------------------
FROM base AS all-in-one

ENV PORT=8000 \
    STREAMLIT_PORT=8501 \
    BACKEND_API_URL=http://localhost:8000 \
    DATABASE_PATH=/app/data/maintenance_history.db

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_dashboard.py"]
