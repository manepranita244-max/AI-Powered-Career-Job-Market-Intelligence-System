"""Central configuration: paths, constants and data-source details.

Everything that a beginner might want to tweak lives here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data source
DATA_URL = (
    "https://raw.githubusercontent.com/DebrajAdhikary5/naukri_com_job_analysis/"
    "main/naukri_com-job_sample.csv"
)
DATA_SHA256 = "56a13321e00b3ab2d61b8a45d4c0b08248f3a79979296d64e2186b50344c0bb2"

RAW_PATH = ROOT / "data" / "raw" / "naukri_com-job_sample.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "jobs_clean.csv.gz"

# -------------------------------------------------------------------- outputs
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

for _p in (PROCESSED_PATH.parent, MODELS_DIR, FIG_DIR, TABLE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- experiments
RANDOM_STATE = 42
TEST_SIZE = 0.20             # share of *companies* held out for the final test
CV_FOLDS = 5                 # GroupKFold folds used on the training part only

# Job-role target: keep role families with at least this many postings.
MIN_ROLE_SAMPLES = 60
EXCLUDED_ROLE_LABELS = ["Other"]   # "Other" is not a real role -> not a target

# Salary sanity limits (INR lakh per annum) used to drop obvious data-entry errors.
MIN_SALARY_LPA = 0.25
MAX_SALARY_LPA = 100.0

# Minimum number of salaried postings required before we trust a salary model /
# salary statistic for a group (role, city, skill ...).
MIN_SALARY_SUPPORT = 30

# A salary model is only called "reliable" in the app if it passes ALL of these
# on the held-out test set (checked automatically in src/salary_model.py).
SALARY_MIN_TRAIN_ROWS = 1000
SALARY_MIN_R2_LOG = 0.50          # R^2 of log-salary
SALARY_MIN_MAE_GAIN_PCT = 25.0    # MAE must be >=25% lower than "global median" baseline
