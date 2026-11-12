"""Project paths and constants.

The values mirror the notebooks, so a model trained there can be scored here:
same processed file, same feature order, same seed.
"""

from pathlib import Path

# src/bank_churn/config/settings.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Churn_Modelling.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "churn_processed.csv"

# Fitted model artefacts produced by study 3 (not created yet)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_xgb.json"
MODEL_METADATA_PATH = ARTIFACT_DIR / "churn_model_metadata.json"

TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]
BINARY_COLUMNS = ["HasCrCard", "IsActiveMember"]
NUMERIC_COLUMNS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                   "EstimatedSalary"]

# Exact column order of the training matrix in notebooks/modelling.ipynb
FEATURE_ORDER = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "ZeroBalance",
    "Geo_France", "Geo_Germany", "Geo_Spain", "Gender_Female", "Gender_Male",
]

GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Female", "Male"]

SEED = 42
TEST_SIZE = 0.2

# Operating cut-off recommended in FINDINGS.md: the cost-optimal point, not the F2 optimum
DECISION_THRESHOLD = 0.30
