"""Project-wide constants.

Anything defined here is defined exactly once. Import it, never redeclare it.
"""

from datetime import datetime

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
# it everywhere.
#
# Corollary: do not build calendar features that assume real dates (holidays,
# Black Friday, month-end). The offset is probably wrong, so those would land
# on the wrong days.
TRANSACTION_DT_REFERENCE = datetime(2017, 12, 1)

# Column names used across modules, so a typo is an ImportError rather than a
# silently empty result.
TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
RAW_TIME_COLUMN = "TransactionDT"
TIMESTAMP_COLUMN = "timestamp"
