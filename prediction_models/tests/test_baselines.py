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
    assert np.allclose(preds, y.mean())


def test_global_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    X, y = Xy
    baseline = GlobalMeanBaseline().fit(X, y.to_numpy())
    path = tmp_path / "global_mean.pkl"
    baseline.save(path)
    restored = GlobalMeanBaseline().load(path)
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


def test_group_mean_uses_group_means(Xy) -> None:
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    # Each prediction must equal the per-Sitio mean from training
    expected_means = y.groupby(X["Sitio"]).mean().to_dict()
    for sitio, expected in expected_means.items():
        rows = X[X["Sitio"] == sitio]
        if rows.empty:
            continue
        preds = baseline.predict(rows.head(5))
        assert np.allclose(preds, expected)


def test_group_mean_unseen_falls_back_to_global(Xy) -> None:
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
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
    assert mae(y_arr, s) < mae(y_arr, g)


def test_group_mean_save_load_roundtrip(Xy, tmp_path) -> None:
    X, y = Xy
    baseline = GroupMeanBaseline("Sitio").fit(X, y.to_numpy())
    path = tmp_path / "group_mean.pkl"
    baseline.save(path)
    restored = GroupMeanBaseline("Sitio").load(path)
    np.testing.assert_allclose(restored.predict(X), baseline.predict(X))


def test_group_mean_invalid_column_raises() -> None:
    df = pd.DataFrame({"foo": ["a", "b"]})
    with pytest.raises(ValueError):
        GroupMeanBaseline("not_a_col").fit(df, np.array([1.0, 2.0]))
