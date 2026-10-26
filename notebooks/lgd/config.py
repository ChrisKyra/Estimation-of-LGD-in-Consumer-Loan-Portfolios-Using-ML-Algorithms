"""Paths, run settings and feature lists shared by every notebook."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Δεδομένα"
RAW_DATA_FILE = DATA_DIR / "Updated.xlsx"                  # raw loan-level data
THESIS_DATA_FILE = DATA_DIR / "Dataset_with_rec_rate.xlsx"  # processed data used for the thesis results

OUTPUT_DIR = Path(os.environ.get("LGD_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
MODEL_DATA_FILE = OUTPUT_DIR / "model_dataset.csv"          # written by notebook 01
for directory in (FIGURE_DIR, PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
SEED = 42

# LGD_QUICK=1 runs every notebook with a handful of trials / epochs so that it
# finishes in a few minutes. Use it to check that everything runs, never for results.
QUICK = os.environ.get("LGD_QUICK") == "1"

N_TRIALS = 3 if QUICK else 40        # Optuna trials per tree model
N_TRIALS_NN = 3 if QUICK else 30     # Optuna trials per neural network
MAX_EPOCHS = 3 if QUICK else 200     # upper bound; early stopping usually ends training much sooner
PATIENCE = 10                        # early-stopping patience (epochs without validation improvement)
CV_FOLDS = 5
N_BOOTSTRAP = 100 if QUICK else 1000
# Hyperparameter searches use at most this many training rows (a stratified sample); the best settings are then
# refitted on all training rows. Only matters for large datasets.
TUNING_SAMPLE = 50_000

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGETS = ["cure", "rec_rate"]
# Stored in the modelling dataset for bookkeeping, never used as features.
NON_FEATURE_COLUMNS = ["loan_id", "split"]

# Known only after the workout of the defaulted loan has ended, so a model that is
# used at the moment of default cannot see them.
POST_DEFAULT_FEATURES = [
    "recoveries",          # gross recovery after charge-off
    "total_pymnt",         # payments received to date
    "total_rec_int",       # interest received to date
    "total_rec_late_fee",  # late fees received to date
    "writeoff_amnt",       # amount written off
    "outdef_healthy",      # borrower is able to repay again (close to the definition of cure)
    "wo_month",            # month of write-off
    "loan_to_wo_days",     # origination -> write-off
    "default_to_wo_days",  # default -> write-off
]

FEATURE_SETS = ("at_default", "thesis")
