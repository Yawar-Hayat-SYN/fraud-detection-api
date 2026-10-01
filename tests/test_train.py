"""Tests for ml.train, end to end on synthetic data: raw -> pipeline -> model."""

import json
import math
import subprocess

import numpy as np
import polars as pl
import pytest

from ml import config
from ml.data import load_features
from ml.metrics import roc_auc
from ml.pipeline import run as build_and_write
from ml.train import _to_matrix, feature_columns, load_model, predict, train
from tests.test_pipeline import point_config_at, write_dataset

# Small data, so let trees split on few rows and keep the run short.
FAST = {"min_child_samples": 5, "num_boost_round": 30}


@pytest.fixture
def features(tmp_path, monkeypatch) -> pl.DataFrame:
    """Synthetic train features written by the real pipeline."""
    write_dataset(tmp_path)
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    build_and_write("train")
    return load_features("train")


@pytest.fixture
def trained(features):
    return train(config=FAST, model_version="test_v1")


# --- Feature selection ---------------------------------------------------------


def test_identifiers_label_entity_keys_and_absolute_time_are_not_features(trained):
    features = trained.metadata["features"]

    for col in ("TransactionID", "isFraud", "uid", "uid_coarse", "TransactionDT", "day", "timestamp"):
        assert col not in features
    # Entity aggregates stay even though the raw keys go.
    for entity in config.AGG_ENTITIES:
        assert f"{entity}_txn_count_prior" in features
    assert {"hour", "dayofweek", "has_block_01", config.RARE_ID_FLAG} <= set(features)


def test_categoricals_recorded_with_training_levels(trained, features):
    levels = trained.metadata["categorical_features"]

    assert set(levels) == set(config.CATEGORICAL_COLS)
    assert levels["card4"] == ["discover", "mastercard", "visa"]
    assert levels["uid_tier"] == sorted(features["uid_tier"].unique().to_list())


def test_lightgbm_splits_categoricals_by_category(trained):
    """card4 carries the planted signal; LightGBM must split it as a category
    ("==" on a set of codes), not as a number ("<=" on a code)."""
    dumped = trained.booster.dump_model()
    names = dumped["feature_names"]

    def splits(node):
        if "split_feature" in node:
            yield names[node["split_feature"]], node["decision_type"]
            yield from splits(node["left_child"])
            yield from splits(node["right_child"])

    card4 = {kind for tree in dumped["tree_info"] for feat, kind in splits(tree["tree_structure"]) if feat == "card4"}
    assert card4 == {"=="}


def test_undeclared_string_column_is_rejected():
    schema = pl.Schema({"TransactionID": pl.Int64, "amount": pl.Float64, "DeviceInfo": pl.String})

    with pytest.raises(ValueError, match="DeviceInfo"):
        feature_columns(schema)


# --- No imputation ---------------------------------------------------------------


def test_nulls_reach_lightgbm_as_nan_not_imputed():
    df = pl.DataFrame(
        {
            "amount": [10.0, None, 0.0],
            "count": [None, 2, 0],
            "flag": [True, None, False],
            "card4": ["visa", None, "amex"],
        }
    )

    X = _to_matrix(df, ["amount", "count", "flag", "card4"], {"card4": ["amex", "visa"]})

    assert np.isnan(X[1, 0]) and X[2, 0] == 0.0, "null and zero must stay distinct"
    assert np.isnan(X[0, 1]) and X[2, 1] == 0.0
    assert np.isnan(X[1, 2]) and X[2, 2] == 0.0
    assert X[0, 3] == 1.0 and X[2, 3] == 0.0 and np.isnan(X[1, 3])


def test_training_data_still_has_nulls(features, trained):
    X = _to_matrix(features, trained.metadata["features"], trained.metadata["categorical_features"])

    assert np.isnan(X).any(), "synthetic features contain nulls; they should reach LightGBM as NaN"


# --- Saved artefacts --------------------------------------------------------------


def test_writes_model_and_complete_metadata(trained):
    out = config.MODELS_DIR / "test_v1"
    assert (out / "model.txt").exists()
    meta = json.loads((out / "metadata.json").read_text())

    assert meta == trained.metadata
    assert meta["model_version"] == "test_v1"
    assert meta["split"] == "train"
    assert meta["split_version"] == config.SPLIT_VERSION
    assert meta["n_rows"] == json.loads(config.SPLIT_PATH.read_text())["train"]["rows"]
    assert meta["hyperparameters"]["min_child_samples"] == 5
    assert meta["num_boost_round"] == 30
    assert "trained_at" in meta and meta["features"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    assert meta["git_commit"] == head.stdout.strip()
    assert not list(config.MODELS_DIR.glob(".*.tmp")), "temporary directory left behind"


def test_versions_are_immutable(trained):
    with pytest.raises(FileExistsError, match="immutable"):
        train(config=FAST, model_version="test_v1")

    replaced = train(config=FAST, model_version="test_v1", overwrite=True)
    assert replaced.metadata["trained_at"] != trained.metadata["trained_at"]


@pytest.mark.parametrize("bad", ["", "../escape", "v 1", "a/b"])
def test_rejects_unsafe_version_names(features, bad):
    with pytest.raises(ValueError, match="model_version"):
        train(config=FAST, model_version=bad)


def test_training_is_reproducible(features):
    a = train(config=FAST, model_version="det_a")
    b = train(config=FAST, model_version="det_b")

    assert a.booster.model_to_string() == b.booster.model_to_string()


# --- Loading and scoring ---------------------------------------------------------


def test_load_and_predict_round_trip(trained, features):
    loaded = load_model("test_v1")
    scores = predict(loaded, features)

    assert loaded.version == "test_v1"
    assert scores.shape == (features.height,)
    assert ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_array_equal(scores, predict(trained, features))


def test_model_learns_the_planted_signal(trained, features):
    assert roc_auc(features[config.TARGET_COL], predict(trained, features)) > 0.8


def test_predict_ignores_column_order_and_extra_columns_and_accepts_lazy(trained, features):
    expected = predict(trained, features)
    shuffled = features.select(reversed(features.columns)).with_columns(pl.lit(1).alias("unrelated"))

    np.testing.assert_array_equal(predict(trained, shuffled), expected)
    np.testing.assert_array_equal(predict(trained, features.lazy()), expected)


def test_unseen_category_is_treated_as_missing(trained, features):
    unseen = features.with_columns(pl.lit("never-seen-card").alias("card4"))
    missing = features.with_columns(pl.lit(None, dtype=pl.String).alias("card4"))

    np.testing.assert_array_equal(predict(trained, unseen), predict(trained, missing))


def test_predict_rejects_missing_features(trained, features):
    with pytest.raises(ValueError, match="missing model features"):
        predict(trained, features.drop("card4"))


def test_load_missing_model_names_command(features):
    with pytest.raises(FileNotFoundError, match="python -m ml.train --version nope"):
        load_model("nope")


# --- Repo hygiene ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "ignored"),
    [("models/lgbm_v1/model.txt", True), ("models/lgbm_v1/metadata.json", False), ("models/x.bin", True)],
)
def test_gitignore_keeps_only_metadata(path, ignored):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=config.REPO_ROOT)
    assert (result.returncode == 0) == ignored


def test_no_nan_scores(trained, features):
    assert not any(math.isnan(s) for s in predict(trained, features))
