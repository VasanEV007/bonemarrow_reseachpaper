"""
app.py - FastAPI Backend Service for BMT Survival Prediction
Provides RESTful prediction endpoints, model metadata, dynamic schemas,
evaluation chart payloads, and CSV exports.
Mounts frontend static assets for single-command deployment.
"""

import os
# Prevent Windows OpenBLAS thread allocation memory issues
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .model_service import service
from .schemas import (
    HybridCascadePredictionResponse,
    LinearRegressionPredictionResponse,
    ModelComparisonResponse,
    PatientInputRecord,
)

app = FastAPI(
    title="Bone Marrow Transplant Survival Prediction API",
    description="Research-based ML prediction service implementing Linear Regression and PNN-Random Forest Cascade Classifier.",
    version="1.0.0",
)

# Enable CORS for external frontend consumers if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Returns application status and model availability."""
    return {
        "status": "online" if service.is_ready else "initializing",
        "service": "BMT Survival Prediction API",
        "models_loaded": {
            "linear_regression": service.lr_model is not None,
            "cascade_classifier": service.cascade_model is not None,
            "preprocessor": service.preprocessor is not None,
        },
        "target_variables": {
            "model_1": "survival_time (continuous, days)",
            "model_2": "survival_status (binary, 0=Survived, 1=Deceased)",
        },
        "features_count": len(service.get_schema()["features"]),
    }


@app.get("/api/schema")
async def get_schema():
    """Returns the 35 verified patient features, section groupings, ranges, and defaults."""
    return service.get_schema()


@app.get("/api/models")
async def get_models():
    """Returns models metadata, target definitions, and verified metrics."""
    return service.get_models_info()


@app.post("/api/predict/linear-regression", response_model=LinearRegressionPredictionResponse)
async def predict_linear_regression(record: PatientInputRecord):
    """Predicts continuous survival_time (in days) using Model 1."""
    try:
        return service.predict_linear_regression(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Linear Regression prediction failed: {str(e)}")


@app.post("/api/predict/hybrid", response_model=HybridCascadePredictionResponse)
async def predict_hybrid(record: PatientInputRecord):
    """Predicts binary survival_status (Survived vs Deceased) using Model 2 (Algorithm 1)."""
    try:
        return service.predict_hybrid_cascade(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cascade Classifier prediction failed: {str(e)}")


@app.post("/api/predict/compare", response_model=ModelComparisonResponse)
async def predict_compare(record: PatientInputRecord):
    """Executes both models on identical patient inputs and returns side-by-side comparison."""
    try:
        return service.predict_compare(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")


@app.get("/api/metrics")
async def get_metrics():
    """Returns verified model evaluation metrics for both models."""
    return service.metrics_data


@app.get("/api/visualizations")
async def get_visualizations_data():
    """Returns chart datasets for interactive Chart.js graphs."""
    return service.charts_data


# -----------------------------------------------------------------------------
# CSV EXPORT ENDPOINTS
# -----------------------------------------------------------------------------
@app.post("/api/export/patient-prediction")
async def export_patient_prediction(payload: Dict[str, Any]):
    """Generates and downloads patient_prediction.csv."""
    record_dict = payload.get("inputs", {})
    pred_data = payload.get("prediction", {})
    csv_str = service.export_patient_prediction_csv(record_dict, pred_data)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=patient_prediction.csv"},
    )


@app.get("/api/export/metrics")
async def export_metrics_csv():
    """Downloads model_evaluation_metrics.csv."""
    csv_str = service.export_metrics_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=model_evaluation_metrics.csv"},
    )


@app.get("/api/export/test-regression")
async def export_test_regression_csv():
    """Downloads regression_test_predictions.csv."""
    csv_str = service.export_test_regression_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=regression_test_predictions.csv"},
    )


@app.get("/api/export/test-classification")
async def export_test_classification_csv():
    """Downloads classification_test_predictions.csv."""
    csv_str = service.export_test_classification_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=classification_test_predictions.csv"},
    )


@app.get("/api/export/feature-importance")
async def export_feature_importance_csv():
    """Downloads feature_importance.csv."""
    csv_str = service.export_feature_importance_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feature_importance.csv"},
    )


# -----------------------------------------------------------------------------
# STATIC FILE SERVING FOR FRONTEND
# -----------------------------------------------------------------------------
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
