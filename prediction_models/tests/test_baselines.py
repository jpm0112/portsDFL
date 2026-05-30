"""Tests for the global-mean and group-mean baselines."""

import numpy as np
import pandas as pd
import pytest  # the test framework; `pytest` runs every `test_*` function below

from ports_dfl.metrics.regression import mae
from ports_dfl.models.baselines import GlobalMeanBaseline, GroupMeanBaseline


# pytest auto-discovers any function named `test_*` and runs it as a test.
# `Xy` here is NOT a normal argument: it is a pytest *fixture* (defined in
# conftest.py). By listing its name as a parameter, this test "requests" it and
# pytest passes in the fixture's return value -- here `(X, y)`: the feature
# DataFrame and the target Series. `-> None` is a type hint: the function
# returns nothing (tests communicate via `assert`, not return values).
def test_global_mean_predicts_training_mean(Xy) -> None:
    # ARRANGE: unpack the fixture into features X and target y.
    X, y = Xy
    # ACT: fit the baseline on the data, then predict for every row.
    # `.to_numpy()` converts the pandas Series y into a plain numpy array.
    baseline = GlobalMeanBaseline().fit(X, y.to_numpy())
    preds = baseline.predict(X)
    # ASSERT: an `assert` makes the test FAIL if the expression is False.
    # The global-mean baseline must predict the single training mean for every
    # row, so every prediction should equal `y.mean()`. `np.allclose` is a
    # float-tolerant "==" (allows tiny rounding differences); it broadcasts the
    # scalar mean against the whole `preds` array.
    assert np.allclose(preds, y.mean())


# `tmp_path` is a built-in pytest fixture: a fresh, empty temporary directory
# (a Path object) unique to this test and auto-deleted afterward. Requesting it
# lets the test write files to disk safely without touching the real repo.
def test_global_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    # A "roundtrip" test: save the model, load it back, and confirm the loaded
    # copy behaves identically to the original.
    X, y = Xy
    baseline = GlobalMeanBaseline().fit(X, y.to_numpy())
    # `tmp_path / "..."` joins a filename onto the temp dir (Path overloads `/`).
    path = tmp_path / "global_mean.pkl"
    baseline.save(path)
    # Create a brand-new (unfitted) instance and load the saved state into it.
    restored = GlobalMeanBaseline().load(path)
    # `np.testing.assert_allclose` is another float-tolerant equality check (like
    # np.allclose, but it RAISES with a helpful diff on mismatch). The restored
    # model must produce the exact same predictions as the original.
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


def test_group_mean_uses_group_means(Xy) -> None:
    X, y = Xy
    # Fit a baseline that predicts the mean target *within each "Sitio" group*.
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    # Each prediction must equal the per-Sitio mean from training
    # Independently re-compute the expected per-Sitio means with pandas, so the
    # test checks the model against a separate "source of truth", not itself.
    # `.to_dict()` -> {sitio_label: mean_target}.
    expected_means = y.groupby(X["Sitio"]).mean().to_dict()
    # Loop over each group and its expected mean (`.items()` yields key/value
    # pairs). For every group, predicting any of its rows must return that mean.
    for sitio, expected in expected_means.items():
        # `X[X["Sitio"] == sitio]` is boolean-mask filtering: keep only rows
        # whose Sitio equals this group.
        rows = X[X["Sitio"] == sitio]
        # Defensive skip: a group taken from `expected_means` always has rows, so
        # this never triggers in practice -- harmless but effectively dead code.
        if rows.empty:
            continue
        # `.head(5)` = first up-to-5 rows; predictions for any rows of a group
        # should all equal that group's training mean.
        preds = baseline.predict(rows.head(5))
        assert np.allclose(preds, expected)


def test_group_mean_unseen_falls_back_to_global(Xy) -> None:
    # Verifies the documented fallback: a group label never seen in training
    # should fall back to the overall (global) training mean, not crash.
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    # `X.iloc[[0]]` selects the first row by position; the double brackets keep
    # it as a 1-row DataFrame (not a Series). `.copy()` avoids mutating X.
    fake_row = X.iloc[[0]].copy()
    # Overwrite its group label with a sentinel that cannot exist in training.
    fake_row["Sitio"] = "__UNSEEN_SITIO__"
    # Predict returns an array; `[0]` grabs the single prediction.
    pred = baseline.predict(fake_row)[0]
    # `pytest.approx` wraps a number so `==` becomes float-tolerant: the
    # prediction must equal the global mean within a small tolerance.
    assert pred == pytest.approx(y.mean())


def test_group_mean_beats_global_on_training(Xy) -> None:
    """Group mean should fit the training data strictly better than global mean."""
    X, y = Xy
    y_arr = y.to_numpy()
    # Fit-and-predict both baselines on the same data (`.fit(...).predict(...)`
    # chains because fit returns self).
    g = GlobalMeanBaseline().fit(X, y_arr).predict(X)
    s = GroupMeanBaseline("Sitio").fit(X, y_arr).predict(X)
    # The group mean conditions on Sitio, so on the TRAINING data it cannot do
    # worse than the global mean -- its error (MAE = mean absolute error) should
    # be strictly lower. Strict `<` is intentional: it would only fail if every
    # group mean equaled the global mean (a degenerate dataset), which would
    # itself be a red flag worth catching.
    assert mae(y_arr, s) < mae(y_arr, g)


def test_group_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    # Same save/load roundtrip idea as the global-mean test, but the group model
    # also has to persist its per-group means dict and the global fallback.
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    path = tmp_path / "group_mean.pkl"
    baseline.save(path)
    restored = GroupMeanBaseline("Sitio").load(path)
    # Loaded model must reproduce the original's predictions exactly (tolerant).
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


# No fixture needed here -- this test builds its own tiny DataFrame.
def test_group_mean_invalid_column_raises() -> None:
    # A DataFrame whose only column is "foo" (so "not_a_col" is absent).
    df = pd.DataFrame({"foo": ["a", "b"]})
    # `with pytest.raises(ValueError):` asserts that the indented code MUST raise
    # a ValueError; the test FAILS if no error (or a different error) is raised.
    # Here fitting on a missing group column should error out clearly.
    with pytest.raises(ValueError):
        GroupMeanBaseline("not_a_col").fit(df, np.array([1.0, 2.0]))
