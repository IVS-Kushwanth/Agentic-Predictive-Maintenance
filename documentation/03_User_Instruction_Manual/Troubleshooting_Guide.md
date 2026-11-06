# System Troubleshooting & Diagnostic Guide
## Agentic Predictive Maintenance Framework

This guide provides troubleshooting solutions for common startup errors, port conflicts, dependency mismatches, database locks, and runtime anomalies.

---

## 1. Quick Diagnostic Checklist

If the system fails to start or behaves unexpectedly, execute these diagnostic checks in sequence:

| Diagnostic Step | Command / Inspection | Expected Result |
| :--- | :--- | :--- |
| **Python Version** | `python --version` | Python `3.10.x`, `3.11.x`, or `3.12.x` (64-bit). |
| **Virtual Environment** | `Get-Command python` (Windows) | Points to `.venv\Scripts\python.exe`. |
| **Model Artifacts** | `Test-Path models\baseline\*.joblib` | Returns `True` (32 models present). |
| **Database File** | `Test-Path data\maintenance_history.db` | Returns `True`. |
| **Port Availability** | `netstat -ano \| findstr :8000` | Port 8000 must not be occupied by another process. |
| **Port Availability** | `netstat -ano \| findstr :8501` | Port 8501 must not be occupied by another process. |

---

## 2. Common Errors and Solutions

### Issue 2.1: Port Conflict (Error: [Errno 10048] address already in use)
- **Symptoms**: FastAPI or Streamlit terminates immediately with `Address already in use` on port 8000 or 8501.
- **Cause**: A previous server instance was terminated abnormally or another application occupies the port.
- **Solution (Windows PowerShell)**:
  ```powershell
  # Find PID occupying port 8000
  $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
  if ($conn) { Stop-Process -Id $conn.OwningProcess -Force; Write-Host "Killed PID $($conn.OwningProcess)" }

  # Find PID occupying port 8501
  $conn = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
  if ($conn) { Stop-Process -Id $conn.OwningProcess -Force; Write-Host "Killed PID $($conn.OwningProcess)" }
  ```
- **Solution (Linux / macOS)**:
  ```bash
  kill -9 $(lsof -t -i:8000) 2>/dev/null
  kill -9 $(lsof -t -i:8501) 2>/dev/null
  ```

---

### Issue 2.2: SQLite Database Locked (Error: sqlite3.OperationalError: database is locked)
- **Symptoms**: Review updates or recommendations fail with `database is locked`.
- **Cause**: Multiple concurrent threads or processes attempted write access without closing transactions.
- **Solution**:
  1. Terminate running server processes (`CTRL+C`).
  2. The recommendation engine implements an automatic 10-second connection timeout:
     ```python
     sqlite3.connect("data/maintenance_history.db", timeout=10.0)
     ```
  3. If corruption persists, backup and vacuum the database:
     ```powershell
     python -c "import sqlite3; conn = sqlite3.connect('data/maintenance_history.db'); conn.execute('VACUUM'); conn.close(); print('DB Vacuumed')"
     ```

---

### Issue 2.3: PyTorch Model Load Error (Error: RuntimeError: Error(s) in loading state_dict)
- **Symptoms**: Prediction agent throws mismatch in layer tensor shapes when loading `.pt` checkpoints.
- **Cause**: Attempting to load an $L=60$ checkpoint into an $L=30$ model instance.
- **Solution**: Ensure the model factory matches the checkpoint window length:
  ```python
  # For L=30 checkpoint:
  model = LSTMRegressor(input_dim=14, hidden_dim=64, num_layers=2)
  model.load_state_dict(torch.load("models/deep_learning/lstm_rul_w30.pt", map_location="cpu"))

  # For L=60 checkpoint:
  model = BiLSTMRegressor(input_dim=14, hidden_dim=64, num_layers=2)
  model.load_state_dict(torch.load("models/deep_learning/bilstm_rul_w60.pt", map_location="cpu"))
  ```

---

### Issue 2.4: Out-of-Range Sensor Values / NaN Inputs (HTTP 422 Unprocessable Content)
- **Symptoms**: API returns HTTP 422 or UI displays red validation alert.
- **Cause**: Negative temperatures (in Kelvin) or non-numeric sensor readings submitted.
- **Solution**: The Pydantic request schema enforces physical bounds (`Air Temp > 250 K`, `Torque > 0 Nm`). Validate telemetry ranges before sending requests.

---

### Issue 2.5: Missing Package Dependencies (ModuleNotFoundError)
- **Symptoms**: `ModuleNotFoundError: No module named 'fastapi'` or similar.
- **Cause**: Dependencies installed outside the active virtual environment.
- **Solution**: Verify active virtual environment path and re-run installation:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
