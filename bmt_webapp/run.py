#!/usr/bin/env python3
"""
Entry point to run the Bone Marrow Transplant Survival Prediction Web Application.
Launches the FastAPI backend and serves the interactive frontend on http://127.0.0.1:8000
"""

import sys
import os

# Prevent Windows OpenBLAS thread allocation memory issues
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import uvicorn

# Ensure the root directory and bmt_webapp directory are in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

def main():
    host = os.environ.get("BMT_HOST", "127.0.0.1")
    port = int(os.environ.get("BMT_PORT", "8000"))
    
    print("=" * 70)
    print(" PEDIATRIC BONE MARROW TRANSPLANT SURVIVAL PREDICTION PLATFORM")
    print(" Research ML System: Ridge Regression & PNN-Random Forest Cascade")
    print("=" * 70)
    print(f"[*] Starting server at: http://{host}:{port}")
    print(f"[*] API Documentation: http://{host}:{port}/docs")
    print(f"[*] Interactive UI:    http://{host}:{port}/")
    print("=" * 70)
    
    uvicorn.run("backend.app:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
