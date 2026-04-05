"""Project configuration constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "encoder.pkl"
METRICS_PATH = ARTIFACTS_DIR / "Metrics.json"

RANDOM_SEED = 42

ACADEMIC_FACTORS_RAW_PATH = RAW_DATA_DIR / "academic-factors.csv"
ACADEMIC_FACTORS_PROCESSED_PATH = PROCESSED_DATA_DIR / "academic_factors_processed.csv"
ACADEMIC_FACTORS_SUMMARY_PATH = PROCESSED_DATA_DIR / "academic_factors_summary.json"

ACADEMIC_FACTORS_TARGET_COLUMN = "Exam_Score"

ACADEMIC_FACTORS_NUMERIC_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score",
]

ACADEMIC_FACTORS_CATEGORICAL_COLUMNS = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

ACADEMIC_FACTORS_REQUIRED_COLUMNS = (
    ACADEMIC_FACTORS_NUMERIC_COLUMNS + ACADEMIC_FACTORS_CATEGORICAL_COLUMNS
)
