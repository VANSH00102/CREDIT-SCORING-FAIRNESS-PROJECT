import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR   = os.path.join(BASE_DIR, "static", "images")

DATASET_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATASET_PATH = os.path.join(DATA_DIR, "german.data")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

PROTECTED_ATTR     = "sex"
PRIVILEGED_GROUP   = 1    # male
UNPRIVILEGED_GROUP = 0    # female
TARGET_COL         = "credit_risk"
DI_THRESHOLD       = 0.80

SECRET_KEY = "creditfair-srm-2024"
PORT       = 5000
DEBUG      = True

COLUMN_NAMES = [
    "checking_account","duration","credit_history","purpose",
    "credit_amount","savings","employment","installment_rate",
    "personal_status","other_debtors","residence_since","property",
    "age","installment_plans","housing","existing_credits",
    "job","dependents","telephone","foreign_worker","credit_risk",
]

CATEGORICAL_COLS = [
    "checking_account","credit_history","purpose","savings",
    "employment","personal_status","other_debtors","property",
    "installment_plans","housing","job","telephone","foreign_worker",
]
NUMERICAL_COLS = [
    "duration","credit_amount","installment_rate",
    "residence_since","age","existing_credits","dependents",
]
