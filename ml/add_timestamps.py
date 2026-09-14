"""Add a real `timestamp` column to both Parquet files and verify the split.

Run from the repo root, after load_raw.py:
    python -m ml.add_timestamps

Uses pyarrow rather than pandas. Loading all 434 columns into a DataFrame
costs 2-3 GB because pandas widens everything to float64, and the copy made
during the transform doubles that -- which is enough to get the process
OOM-killed on an 8 GB machine. We only need one column, so we read the table
in Arrow's own memory format, append a computed column, and write it back.
"""

import gc
import sys
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ml.constants import TRANSACTION_DT_REFERENCE

RAW_DIR = Path("data/raw")


def add_timestamps(name: str) -> dict:
    """Append a timestamp column in place. Returns the time range only.

    Returning the range rather than the table matters: holding both train and
    test in memory at once is what we are trying to avoid.
    """
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        print(f"Missing {path}. Run ml/load_raw.py first.")
        sys.exit(1)

    table = pq.read_table(path)

    if "timestamp" in table.column_names:
        print(f"  {name}: timestamp column already present, replacing it")
        table = table.drop(["timestamp"])

    seconds = table.column("TransactionDT")
    durations = pc.cast(seconds, pa.duration("s"))
    origin = pa.scalar(TRANSACTION_DT_REFERENCE, type=pa.timestamp("s"))
    timestamps = pc.add(origin, durations)

    table = table.append_column("timestamp", timestamps)
    pq.write_table(table, path, compression="snappy")

    lo = pc.min(timestamps).as_py()
    hi = pc.max(timestamps).as_py()
    rows = table.num_rows

    del table, timestamps, durations, seconds
    gc.collect()

    print(f"  {name}: wrote timestamp column, {rows:,} rows")
    return {"min": lo, "max": hi, "rows": rows}


def report(label: str, r: dict) -> None:
    span = (r["max"] - r["min"]).days
    print(f"{label}:")
    print(f"  from {r['min']:%Y-%m-%d %H:%M}")
    print(f"  to   {r['max']:%Y-%m-%d %H:%M}")
    print(f"  span {span} days, {r['rows']:,} rows")


def verify_split(train: dict, test: dict) -> None:
    print("\n" + "=" * 52)
    report("Train", train)
    print()
    report("Test", test)
    print("\n" + "-" * 52)

    if test["min"] <= train["max"]:
        overlap = (train["max"] - test["min"]).days
        print(f"PROBLEM: train and test overlap by {overlap} days.")
        print("The split is not temporal. Stop and investigate before modelling.")
        return

    gap_days = (test["min"] - train["max"]).total_seconds() / 86_400
    print(f"Test begins {gap_days:.0f} days after train ends. No overlap.")
    print()
    print("That gap is part of the task: the model is not predicting tomorrow,")
    print("it is predicting roughly a month past anything it has seen.")
    print("When you build your own validation split out of train.parquet,")
    print("leave a similar gap between the two halves -- otherwise validation")
    print("is an easier problem than the real one and your metrics lie.")


def main() -> None:
    print("Adding timestamps...")
    train = add_timestamps("train")
    test = add_timestamps("test")
    verify_split(train, test)


if __name__ == "__main__":
    main()