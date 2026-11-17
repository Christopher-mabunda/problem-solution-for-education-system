from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "students.csv"

TARGET = "risk"
RISK_LABELS = ["Low","Medium","High"]

FEATURES = [
"attendance",
"assignment_avg",
"test_score",
"study_hours",
"previous_grade",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2