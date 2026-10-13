"""
test_api.py - Automated Test Suite for BMT Survival Prediction Web App
Tests health check, schema endpoints, prediction endpoints for both models,
model comparison mode, input validation, and CSV export functionality.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import unittest
from fastapi.testclient import TestClient
from bmt_webapp.backend.app import app

client = TestClient(app)

SAMPLE_VALID_PATIENT = {
    "Recipientage": 9.6,
    "Recipientgender": 1,
    "Rbodymass": 35.0,
    "Recipientage10": 0,
    "Recipientageint": 1,
    "Donorage": 22.8,
    "Donorage35": 0,
    "Gendermatch": 0,
    "Stemcellsource": 1,
    "DonorABO": 1,
    "RecipientABO": 1,
    "RecipientRh": 1,
    "ABOmatch": 0,
    "CMVstatus": 3,
    "DonorCMV": 1,
    "RecipientCMV": 1,
    "Disease": "all",
    "Diseasegroup": 1,
    "Riskgroup": 1,
    "Txpostrelapse": 0,
    "Relapse": 0,
    "HLAmatch": 0,
    "HLAmismatch": 0,
    "Antigen": -1,
    "Alel": -1,
    "HLAgrI": 0,
    "CD34kgx10d6": 7.2,
    "CD3dCD34": 1.34,
    "CD3dkgx10d8": 5.38,
    "IIIV": 1,
    "aGvHDIIIIV": 0,
    "extcGvHD": 1,
    "ANCrecovery": 19.0,
    "PLTrecovery": 51.0,
    "time_to_aGvHD_III_IV": 32.0,
}


class TestBMTPredictionAPI(unittest.TestCase):

    def test_health_endpoint(self):
        """Verify API health and model availability."""
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["models_loaded"]["linear_regression"])
        self.assertTrue(data["models_loaded"]["cascade_classifier"])
        self.assertEqual(data["features_count"], 35)

    def test_schema_endpoint(self):
        """Verify dynamic schema metadata contains exactly 35 clinical features."""
        response = client.get("/api/schema")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_features"], 35)
        self.assertIn("Recipientage", data["features"])
        self.assertIn("Donorage", data["features"])
        self.assertIn("Disease", data["features"])
        self.assertIn("CD34kgx10d6", data["features"])
        self.assertIn("Demographics", data["sections"])
        self.assertIn("Histocompatibility (HLA)", data["sections"])

    def test_models_metadata_endpoint(self):
        """Verify distinct target descriptions for Model 1 and Model 2."""
        response = client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_1"]["target"], "survival_time")
        self.assertEqual(data["model_1"]["target_units"], "days")
        self.assertEqual(data["model_2"]["target"], "survival_status")

    def test_linear_regression_prediction(self):
        """Verify Model 1 predicts continuous survival_time in days."""
        response = client.post("/api/predict/linear-regression", json=SAMPLE_VALID_PATIENT)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_survival_time_days", data)
        self.assertGreaterEqual(data["predicted_survival_time_days"], 0.0)
        self.assertEqual(data["target_units"], "days")
        self.assertIn("confidence_interval_95_days", data)
        self.assertIn("model_rmse_days", data)
        self.assertIn("model_mae_days", data)

    def test_hybrid_cascade_prediction(self):
        """Verify Model 2 predicts binary survival_status and probabilities."""
        response = client.post("/api/predict/hybrid", json=SAMPLE_VALID_PATIENT)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_class_code", data)
        self.assertIn(data["predicted_class_code"], [0, 1])
        self.assertIn("probability_survived", data)
        self.assertIn("probability_deceased", data)
        self.assertAlmostEqual(data["probability_survived"] + data["probability_deceased"], 1.0, places=2)
        self.assertIn("confidence_percentage", data)

    def test_model_comparison_endpoint(self):
        """Verify simultaneous execution of both models on identical patient inputs."""
        response = client.post("/api/predict/compare", json=SAMPLE_VALID_PATIENT)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_1_regression", data)
        self.assertIn("model_2_classification", data)
        self.assertIn("critical_target_distinction", data)
        self.assertIn("comparison_analysis", data)

    def test_input_validation_rejection(self):
        """Verify API rejects invalid inputs outside physiological range."""
        invalid_patient = dict(SAMPLE_VALID_PATIENT)
        invalid_patient["Recipientage"] = 150.0  # Impossible age for pediatric patient
        response = client.post("/api/predict/linear-regression", json=invalid_patient)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_metrics_and_visualizations_endpoints(self):
        """Verify metric payloads and chart datasets."""
        res_m = client.get("/api/metrics")
        self.assertEqual(res_m.status_code, 200)
        self.assertIn("model_1_regression", res_m.json())
        self.assertIn("model_2_classification", res_m.json())

        res_v = client.get("/api/visualizations")
        self.assertEqual(res_v.status_code, 200)
        v_data = res_v.json()
        self.assertIn("regression_actual_vs_predicted", v_data)
        self.assertIn("classification_roc_curve", v_data)
        self.assertIn("feature_distributions", v_data)

    def test_csv_exports(self):
        """Verify CSV export endpoints return valid text/csv payloads."""
        res_m = client.get("/api/export/metrics")
        self.assertEqual(res_m.status_code, 200)
        self.assertIn("text/csv", res_m.headers["content-type"])

        res_r = client.get("/api/export/test-regression")
        self.assertEqual(res_r.status_code, 200)
        self.assertIn("Sample_ID,Actual_Survival_Time_Days", res_r.text)

        res_c = client.get("/api/export/test-classification")
        self.assertEqual(res_c.status_code, 200)
        self.assertIn("Sample_ID,Actual_Survival_Status", res_c.text)


if __name__ == "__main__":
    unittest.main()
