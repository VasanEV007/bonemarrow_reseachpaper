# Pediatric Bone Marrow Transplant Survival Prediction — Interactive ML Web Application

A full-stack, research-grade clinical machine learning application implementing **Ridge Linear Regression** and the **Cascade-Based Input-Doubling Classifier** (*Izonin et al., 2025*, DOI: [10.3390/computation13040080](https://doi.org/10.3390/computation13040080)) on pediatric allogeneic bone marrow transplantation data.

---

## 1. Research Context & Architecture

This application predicts patient outcomes from **35 verified baseline clinical and donor-recipient attributes** available at the time of transplant, ensuring zero data leakage and preserving clinical temporal validity.

### Critical Target Distinction
The platform hosts two distinct models designed for different clinical prediction questions:
1. **Model 1: Ridge Linear Regression**
   - **Target:** `survival_time` (Continuous duration in days elapsed post-transplant).
   - **Task:** Survival time estimation.
   - **Evaluation Metrics:** RMSE, MAE, R².
2. **Model 2: PNN + Random Forest Cascade Classifier (*Izonin et al., 2025*)**
   - **Target:** `survival_status` (Binary classification: `0 = Survived`, `1 = Deceased`).
   - **Task:** Mortality risk classification.
   - **Stages:**
     1. Median imputation & standard scaling.
     2. Probabilistic Neural Network (PNN) Gaussian kernel density expansion.
     3. Pairwise input doubling feature augmentation ($X \cup X_{\text{pnn}}$).
     4. Random Forest meta-classifier ensemble.
     5. Plurality voting for final discrete class prediction.
   - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, MCC, Cohen's Kappa, ROC AUC.

> **Note:** These models solve distinct mathematical objectives. Their predictions and metrics are **not** interchangeable. The application presents them independently and enables comparative evaluation on identical patient feature inputs.

---

## 2. Directory Structure

```text
bmt_webapp/
├── backend/
│   ├── app.py                  # FastAPI application & REST endpoints
│   ├── model_service.py        # Singleton service for model loading & inference
│   ├── models_core.py          # ML estimators (PNN expander, Preprocessor, Cascade)
│   ├── schemas.py              # Pydantic schemas & 35-attribute clinical metadata
│   └── train_and_export.py     # Training script & test set evaluation exporter
├── frontend/
│   ├── css/
│   │   └── style.css           # Modern dark obsidian/slate scientific CSS design system
│   ├── js/
│   │   ├── app.js              # Application state, 2-way slider sync, inference dispatcher
│   │   ├── charts.js           # Chart.js graphs (Scatter, ROC, Residuals, CM, Distributions)
│   │   └── exports.js          # Safe CSV generation with formula-injection sanitization
│   └── index.html              # Responsive 7-tab dashboard interface
├── models/
│   ├── preprocessor.joblib                         # Fitted scikit-learn pipeline
│   ├── linear_regression_survival_time.joblib       # Fitted Ridge regression model
│   ├── cascade_classifier_survival_status.joblib   # Fitted PNN-RF cascade classifier
│   ├── evaluation_metrics.json                     # Verified held-out test metrics
│   └── evaluation_charts_data.json                 # Pre-computed chart datasets
├── tests/
│   └── test_api.py             # Automated test suite (9 test cases)
├── .gitignore
├── README.md
├── requirements.txt
└── run.py                      # Application launcher script
```

---

## 3. Installation & Quick Start

### Prerequisites
- Python 3.9+ (tested on Python 3.10 – 3.14)
- Pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Application
```bash
python run.py
```
Or directly using Uvicorn:
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### 3. Open in Browser
- **Interactive Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 4. Validated 35 Clinical Input Features

All 35 features are derived from the official Pediatric BMT study schema, organized into 6 clinical sections:

| Clinical Section | Key Attributes Included |
| :--- | :--- |
| **1. Demographics** | `recipient_age`, `donor_age`, `recipient_gender`, `donor_gender`, `gender_match` |
| **2. Disease Classification** | `type` (ALL/AML/Nonmalignant), `disease_group`, `risk_group`, `disease_stage`, `time_to_transplant` |
| **3. Histocompatibility & Matching** | `donor_ABO`, `recipient_ABO`, `ABO_match`, `recipient_rh`, `donor_rh`, `antigen`, `allel`, `HLA_match`, `HLA_mismatch` |
| **4. Transplant & Cell Dosages** | `stem_cell_source`, `CD34kgx10d6` dose, `CD3dCD34` ratio, `CD3kgx10d8` dose |
| **5. Clinical Indicators & Serology** | `donor_recipient_CMV`, `recipient_CMV`, `donor_CMV`, `platelet_recovery` |
| **6. Regimen & Risk Factors** | `conditioning_regimen`, `TBI`, `cyA`, `mtx`, `cyA_plus_mtx`, `cyA_plus_other`, `prednisolone`, `graft_failure` |

*Excluded features:* `id` (database key), `survival_time` (regression target), and `survival_status` (classification target) to eliminate data leakage.

---

## 5. Measured Model Evaluation Benchmarks

Evaluated on the fixed 30-sample held-out test cohort (`fully_filled_30.csv`) after training strictly on the 157-sample training cohort (`remaining_dataset.csv`):

### Model 1: Ridge Linear Regression
- **Test RMSE:** `726.6` days
- **Test MAE:** `562.4` days
- **Test R²:** `-0.024`
- **Context:** Standard linear models experience high variance when predicting exact survival days in pediatric cohorts due to heavy right-censoring.

### Model 2: PNN + Random Forest Cascade Classifier
- **Test Accuracy:** `83.3%` (25 / 30 correct)
- **Test Precision:** `70.0%`
- **Test Recall:** `77.8%`
- **Test F1-Score:** `0.737`
- **Test MCC (Matthews Correlation Coefficient):** `0.627`
- **Test Cohen's Kappa:** `0.615`
- **Test ROC AUC:** `0.820`
- **Confusion Matrix:** True Negatives: 18, False Positives: 3, False Negatives: 2, True Positives: 7.

---

## 6. Running Automated Tests

Run the backend test suite:
```bash
python -m unittest tests/test_api.py
```
All 9 test cases verify:
- Health check and model availability.
- Dynamic schema completeness (35 verified features).
- Metadata endpoints.
- Ridge Linear Regression inference with valid predictions and confidence intervals.
- PNN-RF Cascade inference with class predictions and calibrated probability scores.
- Dual model comparison endpoint.
- Out-of-bounds input validation and missing required fields rejection.
- Static frontend index serving.

---

## 7. CSV Export Center

The application provides CSV exports with formula-injection sanitization (`=`, `+`, `-`, `@` safe prefixes):
1. `patient_prediction.csv`: Submitted inputs, model ID, timestamp, predicted value, confidence interval / probability, and interpretation.
2. `patient_inputs.csv`: Current state of the 35 input fields.
3. `model_evaluation_metrics.csv`: Full benchmark metrics across both models.
4. `regression_test_predictions.csv`: Actual vs predicted survival time and calculated residuals for all 30 test patients.
5. `classification_test_predictions.csv`: Actual vs predicted class and probability scores for all 30 test patients.
6. `classification_confusion_matrix.csv`: 2x2 test confusion matrix breakdown.
7. `feature_importance.csv`: Gini impurity feature ranking for all 35 clinical attributes.

---

## 8. Academic & Ethical Disclaimer

**FOR RESEARCH AND EDUCATIONAL DEMONSTRATION ONLY.**  
This software is an academic demonstration of hybrid machine learning methods on small biomedical datasets. It has **NOT** been evaluated in clinical trials, approved by the FDA, or CE-marked as a medical device. It must **NEVER** be used to guide diagnostic decisions, patient prognoses, or treatment protocols.
