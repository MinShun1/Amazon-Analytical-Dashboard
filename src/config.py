"""Project-wide paths and constants.

Every path is resolved relative to this file, so the app works from any
working directory and on any machine.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "1_Raw"
PROCESSED_DIR = DATA_DIR / "2_Processed"
RAW_ZIP = RAW_DIR / "E-Commerce Sales Dataset.zip"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"

TESTS_DIR = PROJECT_ROOT / "tests"
REFERENCE_VALUES = TESTS_DIR / "reference_values.json"

# File names inside the Kaggle zip (exact names, including the double space).
RAW_FILES = {
    "amazon": "Amazon Sale Report.csv",
    "warehouse": "Cloud Warehouse Compersion Chart.csv",
    "expense": "Expense IIGF.csv",
    "international": "International sale Report.csv",
    "may_2022": "May-2022.csv",
    "pnl_2021": "P  L March 2021.csv",
    "product": "Sale Report.csv",
}

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/thedevastator/"
    "unlock-profits-with-e-commerce-sales-data"
)

# ---------------------------------------------------------------------------
# Analytical constants (taken verbatim from the notebook)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
ALPHA = 0.05

# Section 6 of Data Integration: International main table ends at row 18634.
INTERNATIONAL_MAIN_TABLE_ROWS = 18635
INTERNATIONAL_KEY_COLS = ["date", "customer", "style", "sku", "size"]

MARKETPLACE_COLS = [
    "ajio_mrp",
    "amazon_mrp",
    "amazon_fba_mrp",
    "flipkart_mrp",
    "limeroad_mrp",
    "myntra_mrp",
    "paytm_mrp",
    "snapdeal_mrp",
]

UNRESOLVED_STATUSES = ["Pending", "Pending - Waiting for Pick Up", "Shipping"]

# Experiment 1 temporal split (cell 204)
EXP1_CUTOFF = "2022-05-31"
# Experiment 1.1 temporal split (cells 218 / 227)
EXP11_SPLITS = {
    "train": ("2022-04-01", "2022-04-30"),
    "validation": ("2022-05-01", "2022-05-31"),
    "test": ("2022-06-01", "2022-06-30"),
}
EXP11_TARGET_CANCEL_RATE = 0.35  # notebook comment says "70:30"; code uses 65:35

STATE_MAPPING = {
    "RAJSHTHAN": "RAJASTHAN",
    "ORISSA": "ODISHA",
    "PB": "PUNJAB",
    "RJ": "RAJASTHAN",
    "PUNJAB/MOHALI/ZIRAKPUR": "PUNJAB",
}

INTL_SIZE_GROUPS = {
    "XS": "Standard", "S": "Standard", "M": "Standard", "L": "Standard",
    "XL": "Standard", "XXL": "Standard", "XXXL": "Standard",
    "4XL": "Extended", "5XL": "Extended", "6XL": "Extended",
    "Free": "Free Size",
}

# ---------------------------------------------------------------------------
# Interpreter check (warn only)
# ---------------------------------------------------------------------------
TARGET_PYTHON = (3, 12)

if sys.version_info[:2] != TARGET_PYTHON:
    warnings.warn(
        f"This project targets Python {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]} "
        f"(the notebook's environment); you are running "
        f"{sys.version_info.major}.{sys.version_info.minor}. "
        "Results may differ slightly.",
        RuntimeWarning,
        stacklevel=2,
    )
