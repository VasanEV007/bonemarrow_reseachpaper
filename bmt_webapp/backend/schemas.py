"""
schemas.py - Dynamic Schema & Pydantic Models for BMT Survival Prediction
Defines the clinical schema metadata for the 35 verified patient features
and data models for FastAPI request and response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# CLINICAL FEATURE METADATA DICTIONARY (35 Features)
# -----------------------------------------------------------------------------
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    # Section 1: Patient Demographics & Baseline Characteristics
    "Recipientage": {
        "label": "Recipient Age",
        "section": "Demographics",
        "type": "numeric",
        "min": 0.6,
        "max": 20.2,
        "step": 0.1,
        "default": 9.6,
        "median": 9.6,
        "unit": "years",
        "description": "Age of the pediatric recipient at transplantation time (0.6 to 20.2 years).",
    },
    "Recipientgender": {
        "label": "Recipient Gender",
        "section": "Demographics",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Female (0)"},
            {"value": 1, "label": "Male (1)"},
        ],
        "description": "Biological sex of the recipient.",
    },
    "Rbodymass": {
        "label": "Recipient Body Mass",
        "section": "Demographics",
        "type": "numeric",
        "min": 6.0,
        "max": 103.4,
        "step": 0.5,
        "default": 33.0,
        "median": 33.0,
        "unit": "kg",
        "description": "Weight of the recipient in kilograms at the time of procedure.",
    },
    "Recipientage10": {
        "label": "Recipient Age ≥ 10 Years",
        "section": "Demographics",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Under 10 years (0)"},
            {"value": 1, "label": "10 years or older (1)"},
        ],
        "description": "Binary indicator whether recipient is aged 10 years or older.",
    },
    "Recipientageint": {
        "label": "Recipient Age Category",
        "section": "Demographics",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Early Pediatric (< 5 years) [0]"},
            {"value": 1, "label": "Middle Childhood (5 - 12 years) [1]"},
            {"value": 2, "label": "Adolescent (> 12 years) [2]"},
        ],
        "description": "Discretized pediatric age group interval.",
    },

    # Section 2: Donor Characteristics & Matching
    "Donorage": {
        "label": "Donor Age",
        "section": "Donor & Matching",
        "type": "numeric",
        "min": 18.0,
        "max": 56.0,
        "step": 0.5,
        "default": 33.5,
        "median": 33.5,
        "unit": "years",
        "description": "Age of the hematopoietic stem cell donor in years.",
    },
    "Donorage35": {
        "label": "Donor Age ≥ 35 Years",
        "section": "Donor & Matching",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Donor < 35 years (0)"},
            {"value": 1, "label": "Donor ≥ 35 years (1)"},
        ],
        "description": "Binary indicator whether donor is aged 35 years or older.",
    },
    "Gendermatch": {
        "label": "Gender Match",
        "section": "Donor & Matching",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Mismatch / Other (0)"},
            {"value": 1, "label": "Exact Gender Match (1)"},
        ],
        "description": "Compatibility between donor and recipient biological sex.",
    },
    "Stemcellsource": {
        "label": "Stem Cell Source",
        "section": "Donor & Matching",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Bone Marrow (0)"},
            {"value": 1, "label": "Peripheral Blood Stem Cells (1)"},
        ],
        "description": "Anatomical graft source (Bone Marrow vs. Peripheral Blood).",
    },
    "DonorABO": {
        "label": "Donor ABO Group",
        "section": "Donor & Matching",
        "type": "select",
        "default": 0,
        "options": [
            {"value": -1, "label": "Group O (-1)"},
            {"value": 0, "label": "Group A (0)"},
            {"value": 1, "label": "Group B (1)"},
            {"value": 2, "label": "Group AB (2)"},
        ],
        "description": "ABO blood group classification of the donor.",
    },
    "RecipientABO": {
        "label": "Recipient ABO Group",
        "section": "Donor & Matching",
        "type": "select",
        "default": 0,
        "options": [
            {"value": -1, "label": "Group O (-1)"},
            {"value": 0, "label": "Group A (0)"},
            {"value": 1, "label": "Group B (1)"},
            {"value": 2, "label": "Group AB (2)"},
        ],
        "description": "ABO blood group classification of the recipient.",
    },
    "RecipientRh": {
        "label": "Recipient Rhesus Factor",
        "section": "Donor & Matching",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Rh Negative (0)"},
            {"value": 1, "label": "Rh Positive (1)"},
        ],
        "description": "Rh(D) antigen presence on recipient red blood cells.",
    },
    "ABOmatch": {
        "label": "ABO Match Status",
        "section": "Donor & Matching",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "ABO Incompatible (0)"},
            {"value": 1, "label": "ABO Compatible / Match (1)"},
        ],
        "description": "Immunological compatibility between donor and recipient blood types.",
    },
    "CMVstatus": {
        "label": "CMV Serological Status",
        "section": "Donor & Matching",
        "type": "select",
        "default": 2,
        "options": [
            {"value": 0, "label": "Donor-/Recipient- (0)"},
            {"value": 1, "label": "Donor+/Recipient- (1)"},
            {"value": 2, "label": "Donor-/Recipient+ (2)"},
            {"value": 3, "label": "Donor+/Recipient+ (3)"},
        ],
        "description": "Cytomegalovirus serological match pairing category.",
    },
    "DonorCMV": {
        "label": "Donor CMV Serology",
        "section": "Donor & Matching",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Seronegative (0)"},
            {"value": 1, "label": "Seropositive (1)"},
        ],
        "description": "Donor exposure to cytomegalovirus antibodies.",
    },
    "RecipientCMV": {
        "label": "Recipient CMV Serology",
        "section": "Donor & Matching",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Seronegative (0)"},
            {"value": 1, "label": "Seropositive (1)"},
        ],
        "description": "Recipient pre-existing cytomegalovirus seropositivity.",
    },

    # Section 3: Disease Characteristics & Risk
    "Disease": {
        "label": "Primary Diagnosis (Disease)",
        "section": "Disease & Risk",
        "type": "select",
        "default": "all",
        "options": [
            {"value": "all", "label": "Acute Lymphoblastic Leukemia (ALL)"},
            {"value": "aml", "label": "Acute Myeloid Leukemia (AML)"},
            {"value": "chronic", "label": "Chronic Leukemia / MDS (chronic)"},
            {"value": "nonmalignant", "label": "Non-Malignant Disorder (SAA, FA)"},
            {"value": "lymphoma", "label": "Lymphoma"},
        ],
        "description": "Primary clinical diagnosis indicating allogeneic transplant.",
    },
    "Diseasegroup": {
        "label": "Disease Classification",
        "section": "Disease & Risk",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Non-Malignant (0)"},
            {"value": 1, "label": "Malignant Hematological Disease (1)"},
        ],
        "description": "Broad clinical categorisation into malignant or benign disorders.",
    },
    "Riskgroup": {
        "label": "Clinical Risk Stratification",
        "section": "Disease & Risk",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Standard / Low Risk (0)"},
            {"value": 1, "label": "High Risk (1)"},
        ],
        "description": "Disease risk category assessing cytogenetics and remission status.",
    },
    "Txpostrelapse": {
        "label": "Transplant Post-Relapse",
        "section": "Disease & Risk",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "First Remission / No Relapse (0)"},
            {"value": 1, "label": "Performed After Relapse (1)"},
        ],
        "description": "Indicates whether the transplant was performed following relapse.",
    },
    "Relapse": {
        "label": "Prior Disease Relapse",
        "section": "Disease & Risk",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "No prior relapse (0)"},
            {"value": 1, "label": "Prior relapse documented (1)"},
        ],
        "description": "Documented disease recurrence before conditioning.",
    },

    # Section 4: Histocompatibility & HLA Matching
    "HLAmatch": {
        "label": "HLA Match Degree",
        "section": "Histocompatibility (HLA)",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "10/10 Matched (0)"},
            {"value": 1, "label": "9/10 Matched (1)"},
            {"value": 2, "label": "8/10 Matched (2)"},
            {"value": 3, "label": "7/10 or Lower Matched (3)"},
        ],
        "description": "High-resolution Human Leukocyte Antigen compatibility grade.",
    },
    "HLAmismatch": {
        "label": "HLA Mismatch Indicator",
        "section": "Histocompatibility (HLA)",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Fully Matched (0)"},
            {"value": 1, "label": "Any HLA Mismatch Present (1)"},
        ],
        "description": "Binary flag denoting presence of any HLA incompatibility.",
    },
    "Antigen": {
        "label": "Antigen Match Discrepancy",
        "section": "Histocompatibility (HLA)",
        "type": "select",
        "default": -1,
        "options": [
            {"value": -1, "label": "No antigen differences (-1)"},
            {"value": 0, "label": "0 differences (0)"},
            {"value": 1, "label": "1 antigen difference (1)"},
            {"value": 2, "label": "2 antigen differences (2)"},
        ],
        "description": "Number of antigen-level mismatches detected.",
    },
    "Alel": {
        "label": "Allele Match Discrepancy",
        "section": "Histocompatibility (HLA)",
        "type": "select",
        "default": -1,
        "options": [
            {"value": -1, "label": "No allele differences (-1)"},
            {"value": 0, "label": "0 differences (0)"},
            {"value": 1, "label": "1 allele difference (1)"},
            {"value": 2, "label": "2 allele differences (2)"},
            {"value": 3, "label": "3 allele differences (3)"},
        ],
        "description": "Number of high-resolution allele-level mismatches.",
    },
    "HLAgrI": {
        "label": "HLA Difference Group",
        "section": "Histocompatibility (HLA)",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "Group 0 (Matched) [0]"},
            {"value": 1, "label": "Group 1 [1]"},
            {"value": 2, "label": "Group 2 [2]"},
            {"value": 3, "label": "Group 3 [3]"},
            {"value": 4, "label": "Group 4 [4]"},
            {"value": 5, "label": "Group 5 [5]"},
            {"value": 7, "label": "Group 7 [7]"},
        ],
        "description": "Categorized histocompatibility immunogenetic distance score.",
    },

    # Section 5: Cell Doses & Graft Composition
    "CD34kgx10d6": {
        "label": "CD34+ Cell Dose",
        "section": "Cell Doses & Graft",
        "type": "numeric",
        "min": 0.5,
        "max": 58.0,
        "step": 0.25,
        "default": 9.72,
        "median": 9.72,
        "unit": "×10⁶ cells/kg",
        "description": "Quantified hematopoietic stem cell graft dose per kg recipient body weight.",
    },
    "CD3dCD34": {
        "label": "CD3+ to CD34+ Cell Ratio",
        "section": "Cell Doses & Graft",
        "type": "numeric",
        "min": 0.2,
        "max": 99.5,
        "step": 0.5,
        "default": 2.73,
        "median": 2.73,
        "unit": "ratio",
        "description": "Ratio of T-lymphocytes (CD3+) to stem cells (CD34+) in the infused graft.",
    },
    "CD3dkgx10d8": {
        "label": "CD3+ T-Cell Dose",
        "section": "Cell Doses & Graft",
        "type": "numeric",
        "min": 0.04,
        "max": 20.0,
        "step": 0.1,
        "default": 4.32,
        "median": 4.32,
        "unit": "×10⁸ cells/kg",
        "description": "Absolute CD3+ mature T-lymphocyte cell dose per kg recipient body weight.",
    },

    # Section 6: Recovery Dynamics & GvHD Markers
    "IIIV": {
        "label": "Acute GvHD Grade II–IV",
        "section": "Recovery & GvHD",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "No significant aGvHD (0)"},
            {"value": 1, "label": "Grade II–IV aGvHD observed (1)"},
        ],
        "description": "Clinically diagnosed acute Graft-versus-Host Disease (Grade 2 to 4).",
    },
    "aGvHDIIIIV": {
        "label": "Severe Acute GvHD (Grade III–IV)",
        "section": "Recovery & GvHD",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "No severe aGvHD (0)"},
            {"value": 1, "label": "Severe Grade III–IV aGvHD (1)"},
        ],
        "description": "Severe lifethreatening acute Graft-versus-Host Disease.",
    },
    "extcGvHD": {
        "label": "Extensive Chronic GvHD",
        "section": "Recovery & GvHD",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Limited / No chronic GvHD (0)"},
            {"value": 1, "label": "Extensive chronic GvHD (1)"},
        ],
        "description": "Diagnosis of extensive chronic Graft-versus-Host Disease.",
    },
    "ANCrecovery": {
        "label": "Neutrophil Recovery Time (ANC)",
        "section": "Recovery & GvHD",
        "type": "numeric",
        "min": 9.0,
        "max": 60.0,
        "step": 1.0,
        "default": 15.0,
        "median": 15.0,
        "unit": "days",
        "description": "Time in days until absolute neutrophil count recovery (ANC > 500/μL).",
    },
    "PLTrecovery": {
        "label": "Platelet Recovery Time (PLT)",
        "section": "Recovery & GvHD",
        "type": "numeric",
        "min": 9.0,
        "max": 120.0,
        "step": 1.0,
        "default": 21.0,
        "median": 21.0,
        "unit": "days",
        "description": "Time in days until independent platelet recovery (PLT > 20,000/μL).",
    },
    "time_to_aGvHD_III_IV": {
        "label": "Time to Severe Acute GvHD",
        "section": "Recovery & GvHD",
        "type": "numeric",
        "min": 10.0,
        "max": 100.0,
        "step": 1.0,
        "default": 100.0,
        "median": 100.0,
        "unit": "days",
        "description": "Days post-transplant until onset of severe Grade III–IV aGvHD (100 = censored / no onset).",
    },
}


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS FOR API REQUESTS AND RESPONSES
# -----------------------------------------------------------------------------
class PatientInputRecord(BaseModel):
    """35 Validated Clinical Features for BMT Survival Prediction."""
    Recipientage: float = Field(..., ge=0.0, le=30.0, description="Recipient age in years")
    Recipientgender: int = Field(..., ge=0, le=1)
    Rbodymass: float = Field(..., ge=1.0, le=150.0, description="Body mass in kg")
    Recipientage10: int = Field(..., ge=0, le=1)
    Recipientageint: int = Field(..., ge=0, le=2)

    Donorage: float = Field(..., ge=15.0, le=75.0, description="Donor age in years")
    Donorage35: int = Field(..., ge=0, le=1)
    Gendermatch: int = Field(..., ge=0, le=1)
    Stemcellsource: int = Field(..., ge=0, le=1)
    DonorABO: int = Field(..., ge=-1, le=2)
    RecipientABO: int = Field(..., ge=-1, le=2)
    RecipientRh: int = Field(..., ge=0, le=1)
    ABOmatch: int = Field(..., ge=0, le=1)
    CMVstatus: int = Field(..., ge=0, le=3)
    DonorCMV: int = Field(..., ge=0, le=1)
    RecipientCMV: int = Field(..., ge=0, le=1)

    Disease: str = Field(..., description="Disease type: all, aml, chronic, nonmalignant, lymphoma")
    Diseasegroup: int = Field(..., ge=0, le=1)
    Riskgroup: int = Field(..., ge=0, le=1)
    Txpostrelapse: int = Field(..., ge=0, le=1)
    Relapse: int = Field(..., ge=0, le=1)

    HLAmatch: int = Field(..., ge=0, le=3)
    HLAmismatch: int = Field(..., ge=0, le=1)
    Antigen: int = Field(..., ge=-1, le=3)
    Alel: int = Field(..., ge=-1, le=3)
    HLAgrI: int = Field(..., ge=0, le=7)

    CD34kgx10d6: float = Field(..., ge=0.1, le=80.0, description="CD34+ cell dose")
    CD3dCD34: float = Field(..., ge=0.05, le=150.0, description="CD3 to CD34 ratio")
    CD3dkgx10d8: float = Field(..., ge=0.01, le=30.0, description="CD3+ cell dose")

    IIIV: int = Field(..., ge=0, le=1)
    aGvHDIIIIV: int = Field(..., ge=0, le=1)
    extcGvHD: int = Field(..., ge=0, le=1)
    ANCrecovery: float = Field(..., ge=1.0, le=1000000.0, description="ANC recovery days")
    PLTrecovery: float = Field(..., ge=1.0, le=1000000.0, description="Platelet recovery days")
    time_to_aGvHD_III_IV: float = Field(..., ge=1.0, le=1000000.0, description="Time to aGvHD III-IV days")


class LinearRegressionPredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    model_name: str
    model_type: str = "Linear Regression (Ridge Regularized OLS)"
    target_name: str = "survival_time"
    target_units: str = "days"
    predicted_survival_time_days: float
    confidence_interval_95_days: Dict[str, float]
    model_rmse_days: float
    model_mae_days: float
    model_r2_score: float
    interpretation: str
    disclaimer: str


class HybridCascadePredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    model_name: str
    model_type: str = "PNN + Random Forest Cascade-Based Input-Doubling Classifier"
    research_reference: str
    target_name: str = "survival_status"
    predicted_class_code: int
    predicted_class_label: str
    probability_survived: float
    probability_deceased: float
    confidence_percentage: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1_score: float
    test_mcc: float
    test_cohens_kappa: float
    interpretation: str
    disclaimer: str


class ModelComparisonResponse(BaseModel):
    prediction_id: str
    timestamp: str
    patient_summary: Dict[str, Any]
    model_1_regression: LinearRegressionPredictionResponse
    model_2_classification: HybridCascadePredictionResponse
    comparison_analysis: str
    critical_target_distinction: str
    disclaimer: str
