# 08. SQLite & Persistent Storage Cloud Compatibility Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-08-STORAGE-PERSISTENCE  
**Auditor:** Senior Cloud Database & Systems Architect  
**Audit Date:** November 2026  
**Status:** AUDITED — ARCHITECTURAL STORAGE GUIDANCE PROVIDED  

---

## 1. SQLite Database Architecture

The recommendation history and human review lifecycle are persisted in:
- **Default Database Path:** `data/maintenance_history.db`
- **Driver:** Python native `sqlite3`
- **Schema Management:** `init_db()` in `src/database/database.py` with automatic table creation (`IF NOT EXISTS`) and B-Tree indexes on `machine_id`, `review_status`.

---

## 2. Cloud Ephemeral Storage Analysis

### 2.1 The Container Ephemeral Disk Challenge
Most modern cloud container platforms (Streamlit Community Cloud, Heroku, Render free tier, AWS ECS without EFS, Google Cloud Run) operate on **ephemeral filesystems**:
- Every time a container restarts, crashes, or redeploys a new version, the local disk is restored to the base image state.
- **Consequence**: Any new maintenance recommendations, approved work orders, or engineering review notes written to `data/maintenance_history.db` during container runtime will be permanently wiped upon redeployment!

---

## 3. Cloud Storage Migration & Persistence Options

### Option 1: Persistent Volume Mount (Recommended for Docker / VPS / Railway / Render Disk)
- Attach a persistent SSD volume (e.g. `/mnt/data`) and configure the database path:
  ```bash
  export DATABASE_PATH=/mnt/data/maintenance_history.db
  ```
- **Code Change Required**: Enhance `get_db_connection()` to check `os.environ.get("DATABASE_PATH")`.

### Option 2: Managed PostgreSQL Cloud Database (Enterprise Scalability)
- If the application scales to high-volume multi-user concurrent review, migrate the SQLite schema to PostgreSQL (e.g., Supabase, Neon, AWS RDS, Railway Postgres).
- Because the schema uses standard SQL types (`INTEGER`, `TEXT`, `REAL`), migration requires zero algorithmic logic changes.

---

## 4. File Storage Audit (Uploaded Files & Reports)

| File Category | Storage Mechanism | Cloud Persistence Recommendation |
| :--- | :--- | :--- |
| **Uploaded Telemetry CSVs** | Handled in-memory via Pandas `pd.read_csv(uploaded_file)` | Ephemeral memory is appropriate; no persistent disk writes required. |
| **Model Checkpoints** | Bundled in container image (`models/`) | Read-only static assets; fully persistent across restarts. |
| **Generated CSV Export** | Streamlit `st.download_button` generated on-the-fly | In-memory stream; no disk persistence required. |
