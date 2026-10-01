"""Tests for ml.features.aggregates: point-in-time entity aggregates.

Every aggregate for a row must come only from that entity's transactions
strictly before the row's TransactionDT. A leak here makes validation scores
look BETTER, not worse, so nothing else will flag it. These tests are the
only guard.
"""

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml.config import AMT_COL, TIME_COL
from ml.features.aggregates import add_entity_aggregates

ENTITY = "uid"
P = f"{ENTITY}_"  # column prefix
COUNT, MEAN, STD, MAX = (P + s for s in ("txn_count_prior", "amt_mean_prior", "amt_std_prior", "amt_max_prior"))
SINCE = P + "seconds_since_prev"
C1H, C24H, C7D = (P + s for s in ("txn_count_1h", "txn_count_24h", "txn_count_7d"))
AGG_COLS = [COUNT, MEAN, STD, MAX, SINCE, C1H, C24H, C7D]

HOUR, DAY = 3600, 86400


def frame(rows: list[tuple]) -> pl.DataFrame:
    """rows of (TransactionID, entity, TransactionDT, TransactionAmt)."""
    return pl.DataFrame(
        rows,
        schema={"TransactionID": pl.Int64, ENTITY: pl.String, TIME_COL: pl.Int64, AMT_COL: pl.Float64},
        orient="row",
    )


def by_id(df: pl.DataFrame) -> dict[int, dict]:
    return {r["TransactionID"]: r for r in df.iter_rows(named=True)}


# --- 1. Hand-built frame, every value worked out by hand ----------------------

# Entity a: amounts 10, 30, 20, 40 at t = 0, 30min, 1.5h, 25h.
# Entity b: amounts 50, 70 at t = 1000s and 1000s + ~8.1 days.
# Deliberately not in time order, and the two entities are interleaved.
HAND_ROWS = [
    (4, "a", 90_000, 40.0),
    (5, "b", 1_000, 50.0),
    (2, "a", 1_800, 30.0),
    (6, "b", 701_000, 70.0),
    (1, "a", 0, 10.0),
    (3, "a", 5_400, 20.0),
]

# Columns: count, mean, std, max, seconds_since_prev, 1h, 24h, 7d.
# std is the sample std (ddof=1), so it needs at least two prior amounts.
HAND_EXPECTED = {
    # a's first row: no history at all.
    1: (0, None, None, None, None, 0, 0, 0),
    # Prior: [10] at t=0. One value, so std is null.
    2: (1, 10.0, None, 10.0, 1_800, 1, 1, 1),
    # Prior: [10, 30]. Var = ((10-20)^2 + (30-20)^2) / 1 = 200.
    # 1h window is [5400-3600, 5400) = [1800, 5400): t=1800 is exactly an
    # hour back and counts, t=0 does not.
    3: (2, 20.0, math.sqrt(200), 30.0, 3_600, 1, 2, 2),
    # Prior: [10, 30, 20]. Var = (100 + 100 + 0) / 2 = 100.
    # 24h window is [3600, 90000): only t=5400.
    4: (3, 20.0, 10.0, 30.0, 84_600, 0, 1, 3),
    # b's first row.
    5: (0, None, None, None, None, 0, 0, 0),
    # Prior: [50] at t=1000, 700000s back: outside even the 7d window.
    6: (1, 50.0, None, 50.0, 700_000, 0, 0, 0),
}


def test_hand_built_frame_exact():
    out = by_id(add_entity_aggregates(frame(HAND_ROWS), ENTITY))

    for txn_id, expected in HAND_EXPECTED.items():
        actual = tuple(out[txn_id][c] for c in AGG_COLS)
        assert actual == expected, f"TransactionID {txn_id}"


# --- 2. THE LEAKAGE TEST ------------------------------------------------------


def test_appending_future_rows_does_not_change_past_rows():
    """If adding later transactions changes an earlier row's features, the
    features are reading the future. This is the test that matters most."""
    past = frame(
        [
            (1, "a", 0, 10.0),
            (2, "a", 600, 25.0),
            (3, "b", 700, 5.0),
            (4, "a", 2_000, 15.0),
            (5, None, 2_100, 99.0),
        ]
    )
    future = frame(
        [
            # Same entity, later times, including inside every window of the
            # past rows and with a new maximum amount.
            (6, "a", 2_001, 1_000.0),
            (7, "a", 2_500, 0.5),
            (8, "a", 50 * DAY, 3.0),
            (9, "b", 701, 8.0),
            (10, None, 2_200, 1.0),
            (11, "c", 0, 4.0),  # a new entity, early in time
        ]
    )

    before = add_entity_aggregates(past, ENTITY)
    after = add_entity_aggregates(pl.concat([past, future]), ENTITY)

    assert_frame_equal(after.head(past.height), before)


def test_appended_rows_order_does_not_matter():
    """Future rows inserted before past rows in the frame still change nothing."""
    past = frame([(1, "a", 0, 10.0), (2, "a", 600, 25.0)])
    future = frame([(3, "a", 900, 7.0), (4, "a", 1_200, 9.0)])

    before = add_entity_aggregates(past, ENTITY)
    after = add_entity_aggregates(pl.concat([future, past]), ENTITY)

    assert_frame_equal(after.filter(pl.col("TransactionID").is_in([1, 2])), before)


# --- 3. First transaction -----------------------------------------------------


def test_first_transaction_has_zero_count_and_null_stats():
    out = add_entity_aggregates(frame([(1, "a", 100, 10.0), (2, "a", 200, 0.0)]), ENTITY)
    first = out.row(0, named=True)

    assert first[COUNT] == 0
    for col in (MEAN, STD, MAX, SINCE):
        assert first[col] is None, f"{col} should be null, got {first[col]!r}"
    assert first[C1H] == 0


def test_history_of_zero_is_not_null():
    """A prior amount of 0.0 must give mean 0.0, distinct from no history."""
    out = add_entity_aggregates(frame([(1, "a", 100, 0.0), (2, "a", 200, 5.0)]), ENTITY)

    assert out.row(1, named=True)[MEAN] == 0.0
    assert out.row(1, named=True)[MAX] == 0.0


# --- 4. Ties ------------------------------------------------------------------


def test_tied_rows_do_not_see_each_other():
    out = by_id(
        add_entity_aggregates(
            frame(
                [
                    (1, "a", 0, 4.0),
                    (2, "a", 100, 10.0),
                    (3, "a", 100, 20.0),
                    (4, "a", 100, 30.0),
                    (5, "a", 200, 6.0),
                ]
            ),
            ENTITY,
        )
    )

    # All three rows at t=100 see only the row at t=0, identically.
    for txn_id in (2, 3, 4):
        row = out[txn_id]
        assert row[COUNT] == 1
        assert row[MEAN] == 4.0
        assert row[MAX] == 4.0
        assert row[SINCE] == 100, "previous transaction is the one at t=0, not a tie"
        assert row[C1H] == 1

    # The row after them sees all four earlier rows.
    assert out[5][COUNT] == 4
    assert out[5][MEAN] == 16.0
    assert out[5][MAX] == 30.0
    assert out[5][SINCE] == 100


def test_ties_as_first_transactions():
    out = add_entity_aggregates(frame([(1, "a", 50, 10.0), (2, "a", 50, 20.0)]), ENTITY)

    assert out[COUNT].to_list() == [0, 0]
    assert out[MEAN].to_list() == [None, None]
    assert out[SINCE].to_list() == [None, None]


# --- 5. Null entity -----------------------------------------------------------


def test_null_entity_gets_null_aggregates():
    out = by_id(
        add_entity_aggregates(
            frame(
                [
                    (1, None, 0, 10.0),
                    (2, None, 100, 20.0),
                    (3, "a", 50, 5.0),
                    (4, "a", 150, 7.0),
                ]
            ),
            ENTITY,
        )
    )

    for txn_id in (1, 2):
        for col in AGG_COLS:
            assert out[txn_id][col] is None, f"row {txn_id} {col}"

    # Null-entity rows are not pooled into anyone else's history either.
    assert out[4][COUNT] == 1
    assert out[4][MEAN] == 5.0


# --- Frame handling -----------------------------------------------------------


def test_preserves_row_order_and_input():
    df = frame(HAND_ROWS)
    before_cols = df.columns.copy()

    out = add_entity_aggregates(df, ENTITY)

    assert df.columns == before_cols
    assert out["TransactionID"].to_list() == df["TransactionID"].to_list()
    assert out.columns == df.columns + AGG_COLS


def test_lazy_and_eager_agree():
    eager = add_entity_aggregates(frame(HAND_ROWS), ENTITY)
    lazy = add_entity_aggregates(frame(HAND_ROWS).lazy(), ENTITY)

    assert isinstance(lazy, pl.LazyFrame)
    assert_frame_equal(eager, lazy.collect())


def test_custom_windows():
    out = add_entity_aggregates(frame(HAND_ROWS), ENTITY, windows={"2h": 2 * HOUR})

    assert P + "txn_count_2h" in out.columns
    assert C1H not in out.columns
    # Row 3 at t=5400: window [-1800, 5400) holds t=0 and t=1800.
    assert by_id(out)[3][P + "txn_count_2h"] == 2


@pytest.mark.parametrize("bad", [{"0h": 0}, {"neg": -5}])
def test_rejects_non_positive_windows(bad):
    with pytest.raises(ValueError):
        add_entity_aggregates(frame(HAND_ROWS), ENTITY, windows=bad)


# --- Brute-force cross-check ----------------------------------------------------


def brute_force(rows: list[tuple], windows: dict[str, int]) -> dict[int, dict]:
    """The definition, written as literally as possible: for each row, filter
    to the same entity with an earlier TransactionDT, then compute. O(n^2)."""
    out = {}
    for txn_id, ent, t, amt in rows:
        if ent is None:
            out[txn_id] = dict.fromkeys(AGG_COLS[:5] + [P + f"txn_count_{w}" for w in windows])
            continue
        prior = [(pt, pa) for _, pe, pt, pa in rows if pe == ent and pt < t]
        amts = [pa for _, pa in prior if pa is not None]
        n = len(amts)
        mean = sum(amts) / n if n else None
        std = math.sqrt(sum((a - mean) ** 2 for a in amts) / (n - 1)) if n > 1 else None
        out[txn_id] = {
            COUNT: len(prior),
            MEAN: mean,
            STD: std,
            MAX: max(amts) if amts else None,
            SINCE: t - max(pt for pt, _ in prior) if prior else None,
            **{P + f"txn_count_{w}": sum(t - s <= pt for pt, _ in prior) for w, s in windows.items()},
        }
    return out


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_on_random_data(seed):
    import random

    rng = random.Random(seed)
    # Few entities and few distinct times, so ties and window edges are common.
    rows = [
        (
            i,
            rng.choice(["a", "b", "c", None]),
            rng.choice(range(0, 10 * DAY, HOUR)),
            rng.choice([None, 0.0, round(rng.uniform(1, 500), 2)]),
        )
        for i in range(300)
    ]
    windows = {"1h": HOUR, "24h": DAY, "3d": 3 * DAY}

    actual = by_id(add_entity_aggregates(frame(rows), ENTITY, windows=windows))
    expected = brute_force(rows, windows)

    for txn_id, exp in expected.items():
        for col, value in exp.items():
            got = actual[txn_id][col]
            if value is None or got is None:
                assert got == value, f"row {txn_id} {col}: got {got!r}, expected {value!r}"
            else:
                assert got == pytest.approx(value, rel=1e-9, abs=1e-9), f"row {txn_id} {col}"
