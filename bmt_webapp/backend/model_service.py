"""
model_service.py - Model Loading, Preprocessing & Inference Service
Manages loading of saved joblib pipelines and serving real predictions
for Model 1 (Linear Regression) and Model 2 (Cascade Input-Doubling Classifier).
"""

import datetime
import io
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from .schemas import (
    FEATURE_METADATA,
    HybridCascadePredictionResponse,
    LinearRegressionPredictionResponse,
    ModelComparisonResponse,
    PatientInputRecord,
)
from .models_core import (
    CascadeClassifierAppModel,
    ClinicalBMTPreprocessor,
    PNNProbabilityExpander,
)


class ModelService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "..", "models")

        self.preprocessor: Optional[ClinicalBMTPreprocessor] = None
        self.lr_model = None
        self.cascade_model: Optional[CascadeClassifierAppModel] = None
        self.metrics_data: Dict[str, Any] = {}
        self.charts_data: Dict[str, Any] = {}
        self.is_ready = False

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all serialized joblib pipelines and metric artifacts."""
        try:
            prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
            lr_path = os.path.join(self.models_dir, "linear_regression_survival_time.joblib")
            cas_path = os.path.join(self.models_dir, "cascade_classifier_survival_status.joblib")
            met_path = os.path.join(self.models_dir, "evaluation_metrics.json")
            cha_path = os.path.join(self.models_dir, "evaluation_charts_data.json")

            if not os.path.exists(prep_path) or not os.path.exists(lr_path) or not os.path.exists(cas_path):
                print(f"[ModelService] Artifacts not found. Running train_and_export.py...")
                from .train_and_export import train_and_export
                train_and_export()

            self.preprocessor = joblib.load(prep_path)
            self.lr_model = joblib.load(lr_path)
            self.cascade_model = joblib.load(cas_path)

            if os.path.exists(met_path):
                with open(met_path, "r", encoding="utf-8") as f:
                    self.metrics_data = json.load(f)

            if os.path.exists(cha_path):
                with open(cha_path, "r", encoding="utf-8") as f:
                    self.charts_data = json.load(f)

            self.is_ready = True
            print("[ModelService] All models and preprocessing pipelines successfully loaded.")
        except Exception as e:
            print(f"[ModelService ERROR] Failed to load artifacts: {e}")
            self.is_ready = False

    def get_schema(self) -> Dict[str, Any]:
        """Returns the supported 35 input features, sections, ranges, and defaults."""
        sections = {}
        for feat_name, meta in FEATURE_METADATA.items():
            sec = meta.get("section", "Other")
            if sec not in sections:
                sections[sec] = []
            item = dict(meta)
            item["feature_id"] = feat_name
            sections[sec].append(item)

        return {
            "total_features": len(FEATURE_METADATA),
            "features": FEATURE_METADATA,
            "sections": sections,
        }

    def get_models_info(self) -> Dict[str, Any]:
        """Returns metadata about both models and their distinct targets."""
        return {
            "model_1": {
                "id": "linear_regression",
                "name": "Linear Regression (Ridge Regularized OLS)",
                "type": "Regression",
                "target": "survival_time",
                "target_units": "days",
                "description": "Predicts continuous post-transplantation survival duration in days using baseline clinical attributes.",
                "metrics": self.metrics_data.get("model_1_regression", {}),
            },
            "model_2": {
                "id": "hybrid_cascade",
                "name": "PNN + Random Forest Cascade-Based Input-Doubling Classifier",
                "type": "Binary Classification",
                "target": "survival_status",
                "target_classes": {"0": "Survived", "1": "Deceased"},
                "research_reference": "Izonin et al. (2025), Computation, MDPI. DOI: 10.3390/computation13040080",
                "description": "Predicts binary patient survival status using multi-metric PNN probability expansion, input-doubling pairwise augmentation, and Random Forest plurality voting.",
                "metrics": self.metrics_data.get("model_2_classification", {}),
            },
        }

    def _prepare_input_df(self, record: PatientInputRecord) -> pd.DataFrame:
        """Converts Pydantic patient input record into a single-row DataFrame."""
        return pd.DataFrame([record.model_dump()])

    def predict_linear_regression(self, record: PatientInputRecord) -> LinearRegressionPredictionResponse:
        """Generates continuous survival_time prediction using Model 1."""
        if not self.is_ready:
            raise RuntimeError("Model service is not initialized.")

        input_df = self._prepare_input_df(record)
        X_scaled = self.preprocessor.transform(input_df).values
        pred_days = float(self.lr_model.predict(X_scaled)[0])

        # Clamp physically impossible negative days to 0.0
        pred_days_clamped = max(round(pred_days, 1), 0.0)

        # Margin of uncertainty based on model RMSE (884 days)
        rmse = self.metrics_data.get("model_1_regression", {}).get("rmse", 884.18)
        mae = self.metrics_data.get("model_1_regression", {}).get("mae", 725.61)
        r2 = self.metrics_data.get("model_1_regression", {}).get("r2_score", -0.1498)

        ci_low = max(round(pred_days_clamped - 1.96 * (rmse / np.sqrt(30)), 1), 0.0)
        ci_high = round(pred_days_clamped + 1.96 * (rmse / np.sqrt(30)), 1)

        pred_id = f"PRED-LR-{uuid.uuid4().hex[:8].upper()}"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        interp = (
            f"The Linear Regression model predicts an estimated survival observation duration of "
            f"{pred_days_clamped:.0f} days (~{pred_days_clamped / 365.25:.1f} years). "
            f"Estimated 95% confidence interval for expected cohort mean: [{ci_low:.0f}, {ci_high:.0f}] days."
        )

        return LinearRegressionPredictionResponse(
            prediction_id=pred_id,
            timestamp=ts,
            model_name="Linear Regression (Ridge Regularized OLS)",
            predicted_survival_time_days=pred_days_clamped,
            confidence_interval_95_days={"lower": ci_low, "upper": ci_high},
            model_rmse_days=rmse,
            model_mae_days=mae,
            model_r2_score=r2,
            interpretation=interp,
            disclaimer=(
                "Research demonstration only. Predicted survival time is a statistical model estimate, "
                "not a clinically guaranteed prognosis or medical lifespan diagnosis."
            ),
        )

    def predict_hybrid_cascade(self, record: PatientInputRecord) -> HybridCascadePredictionResponse:
        """Generates binary survival_status prediction using Model 2."""
        if not self.is_ready:
            raise RuntimeError("Model service is not initialized.")

        input_df = self._prepare_input_df(record)
        X_scaled = self.preprocessor.transform(input_df).values

        # Plurality voting probabilities
        probs = self.cascade_model.predict_proba(X_scaled)[0]
        p_surv = float(probs[0])
        p_dec = float(probs[1])

        pred_code = int(self.cascade_model.predict(X_scaled)[0])
        pred_label = "Deceased (1)" if pred_code == 1 else "Survived (0)"
        conf_pct = round(max(p_surv, p_dec) * 100.0, 1)

        m = self.metrics_data.get("model_2_classification", {})
        acc = m.get("accuracy", 0.6333)
        prec = m.get("precision", 0.0)
        rec = m.get("recall", 0.0)
        f1 = m.get("f1_score", 0.0)
        mcc = m.get("mcc", -0.1313)
        kappa = m.get("cohens_kappa", -0.0645)

        pred_id = f"PRED-CAS-{uuid.uuid4().hex[:8].upper()}"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        interp = (
            f"The Cascade-Based Input-Doubling Classifier predicts binary outcome: {pred_label} "
            f"with a plurality voting support probability of {conf_pct:.1f}%. "
            f"(P(Survived) = {p_surv * 100:.1f}%, P(Deceased) = {p_dec * 100:.1f}%)."
        )

        return HybridCascadePredictionResponse(
            prediction_id=pred_id,
            timestamp=ts,
            model_name="PNN + Random Forest Cascade-Based Input-Doubling Classifier",
            research_reference="Izonin et al. (2025), Computation, MDPI. DOI: 10.3390/computation13040080",
            predicted_class_code=pred_code,
            predicted_class_label=pred_label,
            probability_survived=round(p_surv, 4),
            probability_deceased=round(p_dec, 4),
            confidence_percentage=conf_pct,
            test_accuracy=acc,
            test_precision=prec,
            test_recall=rec,
            test_f1_score=f1,
            test_mcc=mcc,
            test_cohens_kappa=kappa,
            interpretation=interp,
            disclaimer=(
                "Research demonstration only. Survival status classification is an algorithmic prototype "
                "based on experimental machine learning and must not be used for patient treatment decisions."
            ),
        )

    def predict_compare(self, record: PatientInputRecord) -> ModelComparisonResponse:
        """Executes both models on identical patient inputs and formats side-by-side comparison."""
        res_lr = self.predict_linear_regression(record)
        res_cas = self.predict_hybrid_cascade(record)

        pred_id = f"COMPARE-{uuid.uuid4().hex[:8].upper()}"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        analysis = (
            f"Model 1 (Linear Regression) estimates continuous survival time at {res_lr.predicted_survival_time_days:.0f} days "
            f"(RMSE: {res_lr.model_rmse_days:.1f} days). "
            f"Model 2 (Cascade Classifier) predicts categorical survival status as {res_cas.predicted_class_label} "
            f"(Confidence: {res_cas.confidence_percentage:.1f}%). "
            "Both models evaluated the exact same 35 clinical attributes simultaneously."
        )

        return ModelComparisonResponse(
            prediction_id=pred_id,
            timestamp=ts,
            patient_summary={
                "Recipient_Age": record.Recipientage,
                "Donor_Age": record.Donorage,
                "Disease": record.Disease,
                "CD34_Dose": record.CD34kgx10d6,
                "HLA_Mismatch": "Yes" if record.HLAmismatch == 1 else "No",
                "Stemcell_Source": "Peripheral Blood" if record.Stemcellsource == 1 else "Bone Marrow",
            },
            model_1_regression=res_lr,
            model_2_classification=res_cas,
            comparison_analysis=analysis,
            critical_target_distinction=(
                "CRITICAL TARGET DISTINCTION: Model 1 solves a continuous REGRESSION task (target: survival_time in days), "
                "while Model 2 solves a binary CLASSIFICATION task (target: survival_status: 0=Survived, 1=Deceased). "
                "Their outputs are not directly interchangeable, and their evaluation metrics reflect distinct problem formulations."
            ),
            disclaimer=(
                "Academic research demonstration for comparative modeling analysis. "
                "Not clinically validated for automated diagnostic or therapeutic decisions."
            ),
        )

    # -------------------------------------------------------------------------
    # CSV EXPORT GENERATORS
    # -------------------------------------------------------------------------
    def export_patient_prediction_csv(self, record_dict: Dict[str, Any], pred_data: Dict[str, Any]) -> str:
        """Constructs CSV string for individual patient prediction."""
        rows = [
            ["Item", "Value"],
            ["Prediction_ID", pred_data.get("prediction_id", "N/A")],
            ["Timestamp", pred_data.get("timestamp", "N/A")],
            ["Model_Name", pred_data.get("model_name", "N/A")],
            ["Target_Name", pred_data.get("target_name", "N/A")],
            ["Predicted_Output", pred_data.get("predicted_survival_time_days", pred_data.get("predicted_class_label", "N/A"))],
            ["Confidence_or_Unit", pred_data.get("target_units", f"{pred_data.get('confidence_percentage', 'N/A')}%")],
            ["--- INPUT FEATURES ---", "---"],
        ]
        for k, v in record_dict.items():
            clean_val = str(v).replace('"', '""')
            rows.append([k, f'"{clean_val}"' if any(c in clean_val for c in [",", "\n"]) else clean_val])

        return "\n".join([f"{r[0]},{r[1]}" for r in rows])

    def export_metrics_csv(self) -> str:
        """Constructs CSV string for model evaluation metrics."""
        m_lr = self.metrics_data.get("model_1_regression", {})
        m_cas = self.metrics_data.get("model_2_classification", {})

        rows = [
            ["Model_Name", "Model_Type", "Target", "Metric_1", "Value_1", "Metric_2", "Value_2", "Metric_3", "Value_3"],
            ["Linear_Regression_OLS", "Regression", "survival_time (days)", "RMSE", str(m_lr.get("rmse", 884.18)), "MAE", str(m_lr.get("mae", 725.61)), "R2", str(m_lr.get("r2_score", -0.1498))],
            ["Hybrid_Cascade_Classifier", "Classification", "survival_status", "Accuracy", str(m_cas.get("accuracy", 0.6333)), "F1_Score", str(m_cas.get("f1_score", 0.0)), "MCC", str(m_cas.get("mcc", -0.1313))],
        ]
        return "\n".join([",".join(r) for r in rows])

    def export_test_regression_csv(self) -> str:
        """Constructs CSV string for regression test set predictions."""
        data = self.charts_data.get("regression_actual_vs_predicted", {})
        ids = data.get("sample_ids", [])
        actuals = data.get("actual_time", [])
        preds = data.get("predicted_time", [])
        resids = data.get("residuals", [])

        rows = [["Sample_ID", "Actual_Survival_Time_Days", "Predicted_Survival_Time_Days", "Residual_Days", "Absolute_Error_Days", "Squared_Error_Days2"]]
        for i, a, p, r in zip(ids, actuals, preds, resids):
            abs_err = round(abs(r), 2)
            sq_err = round(r ** 2, 2)
            rows.append([str(i), str(a), str(p), str(r), str(abs_err), str(sq_err)])

        return "\n".join([",".join(r) for r in rows])

    def export_test_classification_csv(self) -> str:
        """Constructs CSV string for classification test set predictions."""
        data = self.charts_data.get("test_classification_results", {})
        ids = data.get("sample_ids", [])
        actuals = data.get("actual_class", [])
        preds = data.get("predicted_class", [])
        probs = data.get("predicted_probability", [])

        rows = [["Sample_ID", "Actual_Survival_Status", "Predicted_Survival_Status", "Predicted_Probability_Deceased", "Prediction_Outcome"]]
        for i, a, p, pr in zip(ids, actuals, preds, probs):
            outcome = "Correct" if a == p else "Incorrect"
            rows.append([str(i), str(a), str(p), str(pr), outcome])

        return "\n".join([",".join(r) for r in rows])

    def export_feature_importance_csv(self) -> str:
        """Constructs CSV string for top feature importances."""
        cas_feats = self.charts_data.get("feature_importances", {}).get("cascade_top", [])
        rows = [["Rank", "Feature_Name", "Importance_Score"]]
        for rank, item in enumerate(cas_feats, 1):
            rows.append([str(rank), item.get("feature", ""), str(item.get("importance", 0.0))])
        return "\n".join([",".join(r) for r in rows])


# Singleton service instance
service = ModelService()
