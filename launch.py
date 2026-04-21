#!/usr/bin/env python3
"""
SDWB Launcher — starts Streamlit + API bridge together.
Double-click this file or run: python launch.py
"""
import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent

# ── Step 1: Install dependencies if missing ───────────────────
REQUIRED = ["streamlit", "fastapi", "uvicorn"]

def install_missing():
    import importlib
    missing = []
    check = {"streamlit": "streamlit", "fastapi": "fastapi", "uvicorn": "uvicorn"}
    for pkg, mod in check.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])
        print("  Done.\n")
    else:
        print("  All dependencies present.\n")

print("\n" + "="*56)
print("  SDWB Launcher")
print("="*56)
install_missing()

# ── Step 2: Start API bridge ──────────────────────────────────
print("  Starting API bridge on http://localhost:8001 ...")
api_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app",
     "--host", "0.0.0.0", "--port", "8001", "--reload"],
    cwd=HERE,
)

# ── Step 3: Start Streamlit ───────────────────────────────────
print("  Starting Streamlit on http://localhost:8501 ...")
print("\n  Both servers running. Open SDWB Dashboard.html — it will show ● LIVE.")
print("  Press Ctrl+C to stop everything.\n")
print("="*56 + "\n")

streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     "--server.port", "8501"],
    cwd=HERE,
)

# ── Wait and handle Ctrl+C cleanly ───────────────────────────
try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\n\n  Shutting down...")
    api_proc.terminate()
    streamlit_proc.terminate()
    api_proc.wait()
    streamlit_proc.wait()
    print("  Done. Goodbye.\n")
