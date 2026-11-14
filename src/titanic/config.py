"""Central configuration: paths, seeds and feature lists live in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_FILE = DATA_RAW / "train.csv"
TEST_FILE = DATA_RAW / "test.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_BEST_CV_FILE = MODELS_DIR / "model_best_cv.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
SUBMISSION_FILE = SUBMISSIONS_DIR / "submission.csv"
SUBMISSION_BEST_CV_FILE = SUBMISSIONS_DIR / "submission_best_cv.csv"

TARGET = "Survived"
ID_COL = "PassengerId"
RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 3

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FarePerPerson",
    "FamilySize",
    "TicketGroupSize",
    "SibSp",
    "Parch",
    "WCGroupRate",
]
CATEGORICAL_FEATURES = [
    "Sex",
    "Pclass",
    "Embarked",
    "Title",
    "Deck",
    "FamilyType",
]
BINARY_FEATURES = ["IsAlone", "HasCabin", "IsChild", "WCGroupKnown", "IsWomanOrChild"]
