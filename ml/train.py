"""Train, save, load and score a LightGBM baseline on the pipeline's features.

Run from anywhere, after python -m ml.pipeline --split train:
    python -m ml.train --version lgbm_v1

Writes models/{model_version}/model.txt and metadata.json. model_version goes
into the predictions table, so every score can be traced back to the model,
data split, feature list, hyperparameters and git commit that produced it.
Versions are immutable: training into an existing version is refused unless
explicitly overwritten.

Two deliberate choices:

- Nulls are never imputed. LightGBM learns a direction for missing values at
  every split, and "we never got this value" is different from "this value was
  zero". The null patterns in this dataset are known to carry signal, and
  imputing would erase them. Nulls reach LightGBM as NaN.

- Categorical columns are cast to pl.Enum with the category list fixed at
  training time and saved in metadata.json, not to pl.Categorical. A
  Categorical's integer codes depend on the order values happened to appear in
  a given frame, so train and predict could silently disagree on which code
  means "visa". With a fixed Enum, the same value always gets the same code,
  and a value never seen in training becomes missing.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ml import config as cfg
from ml.data import load_features

logger = logging.getLogger("ml.train")

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class TrainedModel:
    booster: lgb.Booster
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["model_version"]


def model_dir(model_version: str) -> Path:
    return cfg.MODELS_DIR / model_version


def feature_columns(schema: pl.Schema) -> list[str]:
    """Every column except NON_FEATURE_COLS, in schema order.

    Raises if a remaining column is neither numeric/boolean nor a declared
    categorical: an undeclared string or datetime column has no correct
    default, so it must be classified in ml.config before training.
    """
    features = [c for c in schema.names() if c not in cfg.NON_FEATURE_COLS]
    unusable = [
        f"{c} ({schema[c]})"
        for c in features
        if c not in cfg.CATEGORICAL_COLS and not (schema[c].is_numeric() or schema[c] == pl.Boolean)
    ]
    if unusable:
        raise ValueError(
            f"Columns with no numeric encoding: {unusable}. Add them to "
            f"CATEGORICAL_COLS or NON_FEATURE_COLS in ml/config.py."
        )
    return features


def _category_levels(df: pl.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    """Sorted non-null values of each categorical column, as strings."""
    return {c: sorted(df[c].cast(pl.String).drop_nulls().unique().to_list()) for c in columns}


def _to_matrix(df: pl.DataFrame | pl.LazyFrame, features: list[str], levels: dict[str, list[str]]) -> np.ndarray:
    """Model input matrix: columns in `features` order, nulls as NaN.

    Categoricals become their code in the fixed training Enum; values unseen
    in training become NaN (missing), never a code LightGBM has not seen.
    """
    missing = [c for c in features if c not in df.collect_schema()]
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")

    exprs = [
        pl.col(c).cast(pl.String).cast(pl.Enum(levels[c]), strict=False).to_physical().cast(pl.Float64)
        if c in levels
        else pl.col(c).cast(pl.Float64)
        for c in features
    ]
    selected = df.select(exprs)
    if isinstance(selected, pl.LazyFrame):
        selected = selected.collect()
    return selected.to_numpy()


def _git_state() -> tuple[str | None, bool | None]:
    """(HEAD commit SHA, whether tracked files have uncommitted changes)."""

    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=cfg.REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return sha, (None if status is None else bool(status))


def train(
    split: str = "train",
    config: dict | None = None,
    *,
    model_version: str,
    overwrite: bool = False,
) -> TrainedModel:
    """Train on a split's features and save models/{model_version}/.

    config overrides entries of ml.config.LGBM_PARAMS; it may also set
    "num_boost_round". Returns the trained model with its metadata.
    """
    if not _VERSION_PATTERN.match(model_version):
        raise ValueError(f"model_version {model_version!r} must match {_VERSION_PATTERN.pattern}")
    out_dir = model_dir(model_version)
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists. Model versions are immutable: pick a new "
            f"version, or pass overwrite=True (--overwrite) to replace it."
        )

    params = {**cfg.LGBM_PARAMS, **(config or {})}
    num_boost_round = params.pop("num_boost_round", cfg.LGBM_NUM_BOOST_ROUND)

    df = load_features(split)
    features = feature_columns(df.schema)
    categorical = [c for c in features if c in cfg.CATEGORICAL_COLS]
    levels = _category_levels(df, categorical)
    X = _to_matrix(df, features, levels)
    y = df[cfg.TARGET_COL].to_numpy()
    n_rows, n_positive = df.height, int(df[cfg.TARGET_COL].sum())
    del df
    logger.info("training split=%s rows=%d positives=%d features=%d categorical=%d",
                split, n_rows, n_positive, len(features), len(categorical))

    dataset = lgb.Dataset(X, label=y, feature_name=features, categorical_feature=categorical, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=num_boost_round)

    git_commit, git_dirty = _git_state()
    if git_commit is None:
        logger.warning("Could not read the git commit; this model will not be traceable to code.")
    elif git_dirty:
        logger.warning("Uncommitted changes: git_commit %s does not fully describe this code.", git_commit)

    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "split_version": cfg.SPLIT_VERSION,
        "features": features,
        "categorical_features": levels,
        "n_rows": n_rows,
        "n_positive": n_positive,
        "hyperparameters": params,
        "num_boost_round": num_boost_round,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "lightgbm_version": lgb.__version__,
        "polars_version": pl.__version__,
    }

    # Write into a temporary directory and rename at the end, so a crash never
    # leaves a version with a model but no metadata, or the reverse.
    tmp_dir = out_dir.with_name(f".{model_version}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    booster.save_model(str(tmp_dir / "model.txt"))
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    logger.info("saved %s", out_dir)

    return TrainedModel(booster, metadata)


def load_model(model_version: str) -> TrainedModel:
    """Load a saved model and its metadata."""
    path = model_dir(model_version)
    model_file, metadata_file = path / "model.txt", path / "metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"Model {model_version!r} not found in {path}. Train it with:\n"
            f"    python -m ml.train --version {model_version}"
        )
    metadata = json.loads(metadata_file.read_text())
    booster = lgb.Booster(model_file=str(model_file))
    if booster.feature_name() != metadata["features"]:
        raise ValueError(f"{model_file} and {metadata_file} disagree on the feature list")
    return TrainedModel(booster, metadata)


def predict(model: TrainedModel, df: pl.DataFrame | pl.LazyFrame) -> np.ndarray:
    """Fraud scores in [0, 1], one per row of df, in df's row order.

    df needs the model's feature columns; any others are ignored, and column
    order does not matter.
    """
    X = _to_matrix(df, model.metadata["features"], model.metadata["categorical_features"])
    return model.booster.predict(X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help='model version, e.g. "lgbm_v1"')
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing version")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train(args.split, model_version=args.version, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
