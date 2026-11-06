"""
Agentic Predictive Maintenance Dashboard Runner.
Launches the FastAPI backend and Streamlit web dashboard.

Usage:
  python run_dashboard.py           # Starts backend and frontend together
  python run_dashboard.py --backend # Starts FastAPI backend only (port 8000)
  python run_dashboard.py --frontend# Starts Streamlit frontend only (port 8501)
"""

import os
import sys
import subprocess
import time
import argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON_EXE = sys.executable


def run_backend(port=8000):
    print(f"[*] Starting FastAPI backend on http://localhost:{port}...")
    cmd = [PYTHON_EXE, "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", str(port)]
    return subprocess.Popen(cmd, cwd=BASE_DIR)


def run_frontend(port=8501):
    print(f"[*] Starting Streamlit dashboard on http://localhost:{port}...")
    cmd = [PYTHON_EXE, "-m", "streamlit", "run", "app/frontend/main.py", "--server.port", str(port), "--server.headless", "true"]
    return subprocess.Popen(cmd, cwd=BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Dashboard Launcher")
    parser.add_argument("--backend", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--frontend", action="store_true", help="Launch Streamlit frontend only")
    parser.add_argument("--api-port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Frontend port (default: 8501)")
    args = parser.parse_args()

    if args.backend and not args.frontend:
        p = run_backend(args.api_port)
        p.wait()
    elif args.frontend and not args.backend:
        p = run_frontend(args.ui_port)
        p.wait()
    else:
        # Launch both
        p_back = run_backend(args.api_port)
        time.sleep(2)
        p_front = run_frontend(args.ui_port)

        print("\n" + "=" * 70)
        print("  AGENTIC PREDICTIVE MAINTENANCE WEB DASHBOARD OPERATIONAL")
        print(f"  • FastAPI Swagger Documentation: http://localhost:{args.api_port}/docs")
        print(f"  • Streamlit Web Dashboard:       http://localhost:{args.ui_port}")
        print("=" * 70 + "\n")

        try:
            p_front.wait()
        except KeyboardInterrupt:
            print("\nShutting down services...")
            p_front.terminate()
            p_back.terminate()


if __name__ == "__main__":
    main()
