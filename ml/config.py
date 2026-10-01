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

# TransactionDT in the IEEE-CIS dataset is a count of seconds from an origin
# Kaggle never published. This anchor is ARBITRARY -- the community guess is
# somewhere around December 2017, but it was never confirmed.
#
# It does not matter. Every quantity the model uses is a DIFFERENCE between
# two times: seconds since a user's previous transaction, count of
# transactions in the last hour, whether a row falls before the split cutoff.
# Shift every timestamp by the same amount and all of those are unchanged.
#
# What DOES matter is that this value never changes and is never redefined
# elsewhere. If feature-building uses one anchor and scoring uses another,
# nothing crashes -- the model just quietly gets worse. Define it here, import
# it everywhere. It must also stay equal to the value the committed splits
# were generated with, or their human-readable cutoff dates stop matching.
#
# Corollary: do not build calendar features that assume real dates (holidays,
# Black Friday, month-end). The offset is probably wrong, so those would land
# on the wrong days.
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

# models/{model_version}/model.txt and metadata.json. Only metadata is in git.
MODELS_DIR = REPO_ROOT / "models"

# Kaggle competition the raw CSVs come from.
KAGGLE_COMPETITION = "ieee-fraud-detection"

# --- Splits -----------------------------------------------------------------

SPLIT_VERSION = "v1"
SPLIT_PATH = SPLITS_DIR / f"{SPLIT_VERSION}.json"

# --- Columns ----------------------------------------------------------------

# Column names used across modules, so a typo is an ImportError rather than a
# silently empty result.

# Columns that together approximate a single cardholder identity.
UID_COMPONENTS = ["card1", "addr1", "D1n"]
UID_COARSE_COMPONENTS = ["card1", "addr1"]

ID_COL = "TransactionID"
PRODUCT_COL = "ProductCD"
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
# Datetime derived from TIME_COL and TRANSACTION_DT_REFERENCE by
# ml.features.time. Always computed, never stored in the raw files, so it can
# never disagree with the current anchor.
TIMESTAMP_COL = "timestamp"
AMT_COL = "TransactionAmt"
D1_COL = "D1"
CARD_COL = "card1"
UID_COL = "uid"
UID_COARSE_COL = "uid_coarse"

# ProductCD value that, unlike every other product, has no identity-table data.
W_PRODUCT = "W"

# --- Null-block analysis ----------------------------------------------------

# A null rate below LOW counts as "always present", above HIGH as "always
# blank". Used to classify null blocks by how they vary across products.
NULL_RATE_LOW = 0.01
NULL_RATE_HIGH = 0.99

# --- Availability flags -----------------------------------------------------

# id_21 is present on ~0.9% of rows with 1.6-2.0x fraud lift within every
# product. Its values are noise; only whether it is present carries signal.
RARE_ID_COL = "id_21"
RARE_ID_FLAG = "has_rare_id_block"

# --- Entity aggregates -----------------------------------------------------

# Look-back windows for ml.features.aggregates, as name -> seconds. Each name
# becomes a column suffix, e.g. uid_txn_count_1h.
AGG_WINDOWS = {"1h": 3600, "24h": SECONDS_PER_DAY, "7d": 7 * SECONDS_PER_DAY}

# Entities ml.pipeline aggregates over, finest first. The full UID is precise
# but sparse (median group size 1, so most UIDs have no history); raw card1 is
# dense but coarse (13,553 values over 590,540 rows -- a card fingerprint, not
# an individual card). uid_coarse sits between. The model needs all three.
AGG_ENTITIES = [UID_COL, UID_COARSE_COL, CARD_COL]

# --- Training ---------------------------------------------------------------

# Low-cardinality string columns, passed to LightGBM as categorical features.
# Raw columns are carried through ml.pipeline into the features file;
# uid_tier is derived there.
CATEGORICAL_RAW_COLS = ["ProductCD", "card4", "card6", "DeviceType", *(f"M{i}" for i in range(1, 10))]
CATEGORICAL_COLS = [*CATEGORICAL_RAW_COLS, "uid_tier"]

# Never model inputs:
# - TransactionID and isFraud: an identifier and the label.
# - uid, uid_coarse: raw entity keys, near-unique strings. Their point-in-time
#   aggregates stay; the keys themselves would let the model memorise entities.
# - TransactionDT, day, timestamp: absolute time. Every validation and
#   production row lies beyond the training range, so splits on these learn
#   drift, not fraud. hour and dayofweek are relative and stay.
NON_FEATURE_COLS = [ID_COL, TARGET_COL, UID_COL, UID_COARSE_COL, TIME_COL, "day", TIMESTAMP_COL]

# Fixed baseline hyperparameters. No tuning: this is a baseline, not a
# submission. Seeded and deterministic so a rerun reproduces the model.
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
LGBM_NUM_BOOST_ROUND = 500

# --- Evaluation -------------------------------------------------------------

# Defaults for ml.metrics.evaluate. The review budget is the share of
# transactions the fraud team can manually review; set both to the real
# operating point once it is known.
REVIEW_BUDGET = 0.01
TARGET_FPR = 0.01
