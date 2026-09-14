"""Load the raw IEEE-CIS CSVs, join transaction and identity tables, save as Parquet.

Run from the repo root:
    python ml/load_raw.py

Expects the four Kaggle CSVs in data/raw/:
    train_transaction.csv, train_identity.csv,
    test_transaction.csv,  test_identity.csv
"""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
EXPECTED_TRAIN_ROWS = 590_540
EXPECTED_TEST_ROWS = 506_691
EXPECTED_FRAUD_RATE = 0.035


def check_files_exist() -> None:
    """Fail early with a clear message if any CSV is missing."""
    required = [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
    ]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if missing:
        print(f"Missing files in {RAW_DIR}/:")
        for name in missing:
            print(f"  - {name}")
        print("\nDownload them from the IEEE-CIS Fraud Detection competition on Kaggle.")
        sys.exit(1)


def normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """test_identity.csv uses id-01 while train_identity.csv uses id_01.

    This is a quirk of the original Kaggle release. Left unhandled, the train
    and test frames end up with different column names and every downstream
    join or model call breaks in a confusing way.
    """
    renamed = {col: col.replace("-", "_") for col in df.columns if "-" in col}
    if renamed:
        print(f"  normalised {len(renamed)} hyphenated identity columns")
    return df.rename(columns=renamed)


def load_split(split: str) -> pd.DataFrame:
    """Read one split's two CSVs and left join them on TransactionID."""
    print(f"\nLoading {split}...")

    transactions = pd.read_csv(RAW_DIR / f"{split}_transaction.csv")
    print(f"  {split}_transaction: {transactions.shape[0]:,} rows x {transactions.shape[1]} cols")

    identity = pd.read_csv(RAW_DIR / f"{split}_identity.csv")
    identity = normalise_identity_columns(identity)
    print(f"  {split}_identity:    {identity.shape[0]:,} rows x {identity.shape[1]} cols")

    rows_before = len(transactions)

    merged = transactions.merge(identity, on="TransactionID", how="left")

    # A left join must never change the row count. If it does, TransactionID
    # is not unique in the identity table and rows have been silently
    # duplicated -- which would quietly corrupt every model trained on it.
    if len(merged) != rows_before:
        raise AssertionError(
            f"Join changed row count: {rows_before:,} -> {len(merged):,}. "
            f"TransactionID is not unique in {split}_identity."
        )

    matched = merged["id_01"].notna().sum()
    pct = matched / len(merged) * 100
    print(f"  joined: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    print(f"  {matched:,} rows ({pct:.1f}%) have identity data; the rest are null by design")

    return merged


def sanity_check_train(df: pd.DataFrame) -> None:
    """Confirm the data matches the known shape of this dataset."""
    print("\nSanity checks:")

    rows = len(df)
    print(f"  rows: {rows:,} (expected ~{EXPECTED_TRAIN_ROWS:,})")
    if rows != EXPECTED_TRAIN_ROWS:
        print("  WARNING: row count does not match the published dataset")

    fraud_rate = df["isFraud"].mean()
    fraud_count = int(df["isFraud"].sum())
    print(f"  fraud: {fraud_count:,} of {rows:,} = {fraud_rate:.2%} (expected ~3.5%)")
    if abs(fraud_rate - EXPECTED_FRAUD_RATE) > 0.005:
        print("  WARNING: fraud rate is off -- check you loaded the right file")

    span_days = (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86_400
    print(f"  time span: {span_days:.0f} days of TransactionDT")

    null_pct = df.isna().mean().mean()
    print(f"  overall null rate: {null_pct:.1%} (this dataset is genuinely sparse)")


def save_parquet(df: pd.DataFrame, name: str) -> None:
    """Write to Parquet and report the size saving against the source CSV."""
    out_path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    csv_mb = (RAW_DIR / f"{name}_transaction.csv").stat().st_size / 1024**2
    parquet_mb = out_path.stat().st_size / 1024**2
    print(f"\n  wrote {out_path}")
    print(f"  {csv_mb:.0f} MB CSV -> {parquet_mb:.0f} MB Parquet ({csv_mb / parquet_mb:.1f}x smaller)")


def main() -> None:
    check_files_exist()

    train = load_split("train")
    sanity_check_train(train)
    save_parquet(train, "train")

    test = load_split("test")
    print(f"\n  test rows: {len(test):,} (expected ~{EXPECTED_TEST_ROWS:,})")
    save_parquet(test, "test")

    print("\nDone. Load the result with:")
    print('    pd.read_parquet("data/raw/train.parquet")')


if __name__ == "__main__":
    main()
