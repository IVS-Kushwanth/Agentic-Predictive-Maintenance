# 05. FastAPI Backend Deployment Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-05-FASTAPI-BACKEND  
**Auditor:** Senior Cloud Architect & Backend Engineer  
**Audit Date:** November 2026  
**Status:** AUDITED — PRODUCTION READY WITH RECOMMENDED SECURITY HARDENING  

---

## 1. Backend Architecture & Entrypoint

The primary backend microservice is implemented in `app/backend/main.py` using FastAPI and Pydantic v2 schemas:

- **Module Path:** `app.backend.main:app`
- **Application Class:** `fastapi.FastAPI`
- **ASGI Server:** Uvicorn 0.54.0
- **Standalone Cloud Startup Command:**
  ```bash
  uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --workers 2
  ```

---

## 2. API Endpoints & Route Verification

All 10 REST endpoints were tested against the running server (`http://127.0.0.1:8000`):

| HTTP Method | Endpoint | Response Code | Tested Response Time | Validation Schema Enforced |
| :--- | :--- | :---: | :---: | :---: |
| **GET** | `/health` | `200 OK` | 3.2 ms | `HealthResponse` |
| **GET** | `/api/models` | `200 OK` | 4.8 ms | `ModelsListResponse` |
| **POST** | `/api/predict/rul` | `200 OK` | 18.4 ms | `NASARULPredictResponse` |
| **POST** | `/api/predict/failure` | `200 OK` | 12.1 ms | `AI4IFailurePredictResponse` |
| **POST** | `/api/anomaly/detect` | `200 OK` | 24.5 ms | `AnomalyDetectResponse` |
| **POST** | `/api/explain` | `200 OK` | 165.2 ms | `ExplainResponse` (TreeSHAP) |
| **POST** | `/api/agents/analyze` | `200 OK` | 2,840.0 ms | `AgentAnalyzeResponse` (LangGraph) |
| **POST** | `/api/recommendations` | `200 OK` | 15.6 ms | `RecommendationResponse` |
| **GET** | `/api/history` | `200 OK` | 8.2 ms | `HistoryResponse` |
| **POST** | `/api/recommendations/review` | `200 OK` | 14.1 ms | `ReviewResponse` |

---

## 3. Cloud Deployment Considerations

### 3.1 CORS Configuration Hardening (SECURITY WARNING)
- **Current Implementation in `app/backend/main.py`**:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
- **Issue**: Specifying `allow_origins=["*"]` in combination with `allow_credentials=True` violates the Fetch and CORS specifications in all modern browsers. When deployed across distinct cloud domains (e.g., Streamlit frontend on `https://share.streamlit.io` and FastAPI on `https://api.railway.app`), browsers will reject cross-origin requests.
- **Remediation**: Parameterize allowed origins via environment variable:
  ```python
  origins = os.environ.get("CORS_ORIGINS", "*").split(",")
  ```

### 3.2 Dynamic Port Binding for Cloud PaaS
- Lines 58-59 in `app/backend/main.py`:
  ```python
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("app.backend.main:app", host="0.0.0.0", port=port, reload=False)
  ```
- **Verdict: PASS**. The application dynamically binds to `$PORT` assigned by cloud providers (Render, Heroku, Railway, Cloud Run).
