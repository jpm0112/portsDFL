"""Tests for the global-mean and group-mean baselines."""

import numpy as np
import pandas as pd
import pytest

from ports_dfl.metrics.regression import mae
from ports_dfl.models.baselines import GlobalMeanBaseline, GroupMeanBaseline


def test_global_mean_predicts_training_mean(Xy) -> None:
    X, y = Xy
    baseline = GlobalMeanBaseline().fit(X, y.to_numpy())
    preds = baseline.predict(X)
    # Every prediction must equal the single training mean.
    assert np.allclose(preds, y.mean())


def test_global_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    X, y = Xy
    baseline = GlobalMeanBaseline().fit(X, y.to_numpy())
    path = tmp_path / "global_mean.pkl"
    baseline.save(path)
    restored = GlobalMeanBaseline().load(path)
    # Restored model must reproduce the original's predictions exactly.
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


def test_group_mean_uses_group_means(Xy) -> None:
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    # Re-compute expected per-Sitio means independently, so the test checks the
    # model against a separate source of truth, not itself.
    expected_means = y.groupby(X["Sitio"]).mean().to_dict()
    for sitio, expected in expected_means.items():
        rows = X[X["Sitio"] == sitio]
        # A group drawn from expected_means always has rows; defensive guard.
        if rows.empty:
            continue
        preds = baseline.predict(rows.head(5))
        assert np.allclose(preds, expected)


def test_group_mean_unseen_falls_back_to_global(Xy) -> None:
    # Documented fallback: an unseen group label uses the global mean, not crash.
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    # Double brackets keep this as a 1-row DataFrame; .copy() avoids mutating X.
    fake_row = X.iloc[[0]].copy()
    fake_row["Sitio"] = "__UNSEEN_SITIO__"
    pred = baseline.predict(fake_row)[0]
    assert pred == pytest.approx(y.mean())


def test_group_mean_beats_global_on_training(Xy) -> None:
    """Group mean should fit the training data strictly better than global mean."""
    X, y = Xy
    y_arr = y.to_numpy()
    g = GlobalMeanBaseline().fit(X, y_arr).predict(X)
    s = GroupMeanBaseline("Sitio").fit(X, y_arr).predict(X)
    # Conditioning on Sitio cannot do worse than the global mean on TRAINING
    # data. Strict `<` only fails if every group mean equals the global mean (a
    # degenerate dataset) — itself a red flag worth catching.
    assert mae(y_arr, s) < mae(y_arr, g)


def test_group_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    # Group model must also persist its per-group means dict and global fallback.
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    path = tmp_path / "group_mean.pkl"
    baseline.save(path)
    restored = GroupMeanBaseline("Sitio").load(path)
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


def test_group_mean_invalid_column_raises() -> None:
    df = pd.DataFrame({"foo": ["a", "b"]})
    # Fitting on a missing group column should error out clearly.
    with pytest.raises(ValueError):
        GroupMeanBaseline("not_a_col").fit(df, np.array([1.0, 2.0]))
