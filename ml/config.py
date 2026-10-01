"""Paths and constants for the ml/ package.

This module is the single source of truth for every path and magic number
used in ml/. Nothing else in ml/ should hardcode a path or a magic number:
import it from here instead. If you need a new one, add it here first.

Constants and paths only. No logic, no I/O. Importing this module must never
touch the filesystem.
"""

from datetime import datetime
from pathlib import Path

# --- Time -------------------------------------------------------------------

# TransactionDT is a count of seconds from a reference point the dataset never
# states. This anchor is therefore ARBITRARY: only relative time (differences
# between two TransactionDT values) carries meaning. Do not build features that
# assume real calendar dates (holidays, weekdays, month-end).
#
# Must stay equal to the value the committed splits were generated with, or
# their human-readable cutoff dates stop matching.
TRANSACTION_DT_REFERENCE = datetime(2017, 12, 1)

SECONDS_PER_DAY = 86400

# --- Paths ------------------------------------------------------------------

# Resolved from this file's location, not the cwd, so scripts and tests work
# no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

NULL_BLOCKS_PATH = DATA_DIR / "null_blocks.json"
TRAIN_PARQUET = RAW_DIR / "train.parquet"
TEST_PARQUET = RAW_DIR / "test.parquet"

# Kaggle competition the raw CSVs come from.
KAGGLE_COMPETITION = "ieee-fraud-detection"

# --- Splits -----------------------------------------------------------------

SPLIT_VERSION = "v1"
SPLIT_PATH = SPLITS_DIR / f"{SPLIT_VERSION}.json"

# --- Columns ----------------------------------------------------------------

# Columns that together approximate a single cardholder identity.
UID_COMPONENTS = ["card1", "addr1", "D1n"]
UID_COARSE_COMPONENTS = ["card1", "addr1"]

ID_COL = "TransactionID"
PRODUCT_COL = "ProductCD"
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

# ProductCD value that, unlike every other product, has no identity-table data.
W_PRODUCT = "W"

# --- Null-block analysis ----------------------------------------------------

# A null rate below LOW counts as "always present", above HIGH as "always
# blank". Used to classify null blocks by how they vary across products.
NULL_RATE_LOW = 0.01
NULL_RATE_HIGH = 0.99
