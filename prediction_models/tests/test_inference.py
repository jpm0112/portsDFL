"""Tests for ports_dfl.inference — load_bundle and predict_csv.

A self-contained artifact bundle (preprocessor + one XGBoost model) is built
inside a pytest tmp_path so no pre-trained files are needed on disk.  All
tests are deterministic (seed=42, fixed 200-row slice).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ports_dfl.config import ALL_FEATURES, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.models.xgb import XGBoostRegressorModel


# ---------------------------------------------------------------------------
# Session-scoped artifact bundle — built once, reused across inference tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, Xy) -> Path:
    """Write a minimal real artifact bundle to a temp directory.

    Uses 200 rows, n_estimators=20, no validation set (no early stopping),
    so the bundle is cheap to build and fully deterministic.
    """
    X, y = Xy
    X_small = X.iloc[:200]
    y_small = y.iloc[:200].to_numpy()

    arts = tmp_path_factory.mktemp("artifacts")

    # Preprocessor
    pre = build_preprocessor(categorical_strategy="target")
    Xt = pre.fit_transform(X_small, y_small).astype(np.float32)
    joblib.dump(pre, arts / "preprocessor.pkl")

    # Model — fit without a val set so early stopping is skipped
    m = XGBoostRegressorModel(input_dim=Xt.shape[1], n_estimators=20, random_state=SEED)
    m.fit(Xt, y_small)
    m.save(arts / "xgb.pkl")

    # Manifest fragment
    (arts / "xgb.meta.json").write_text(
        json.dumps(
            {
                "name": "xgb",
                "kind": "tree",
                "artifact": "xgb.pkl",
                "preprocessor": "preprocessor.pkl",
            }
        ),
        encoding="utf-8",
    )

    return arts


# ---------------------------------------------------------------------------
# load_bundle tests
# ---------------------------------------------------------------------------

def test_load_bundle_returns_correct_types(artifacts) -> None:
    """load_bundle returns (preprocessor, dict-of-models, list-of-metas)."""
    from ports_dfl.inference import load_bundle

    pre, models, metas = load_bundle(artifacts)
    assert hasattr(pre, "transform"), "preprocessor must have a transform method"
    assert isinstance(models, dict)
    assert isinstance(metas, list)


def test_load_bundle_loads_xgb_model(artifacts) -> None:
    """The returned dict contains exactly the 'xgb' model from the bundle."""
    from ports_dfl.inference import load_bundle
    from ports_dfl.models.xgb import XGBoostRegressorModel

    _, models, _ = load_bundle(artifacts)
    assert "xgb" in models
    assert isinstance(models["xgb"], XGBoostRegressorModel)


def test_load_bundle_missing_preprocessor_raises(tmp_path) -> None:
    """load_bundle raises FileNotFoundError when preprocessor.pkl is absent."""
    from ports_dfl.inference import load_bundle

    # Write a meta.json but no preprocessor
    (tmp_path / "xgb.meta.json").write_text(
        json.dumps({"name": "xgb", "kind": "tree", "artifact": "xgb.pkl", "preprocessor": "preprocessor.pkl"}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


def test_load_bundle_no_fragments_raises(tmp_path) -> None:
    """load_bundle raises FileNotFoundError when no *.meta.json files exist."""
    from ports_dfl.inference import load_bundle

    # Only a preprocessor, nothing else
    joblib.dump(object(), tmp_path / "preprocessor.pkl")
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


def test_load_bundle_requested_model_absent_raises(artifacts) -> None:
    """Requesting a model not in the bundle raises FileNotFoundError."""
    from ports_dfl.inference import load_bundle

    with pytest.raises(FileNotFoundError):
        load_bundle(artifacts, models=["nope"])


# ---------------------------------------------------------------------------
# predict_csv tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def input_csv(tmp_path_factory, Xy) -> Path:
    """Write a 10-row CSV containing all required feature columns."""
    X, _ = Xy
    csv_path = tmp_path_factory.mktemp("csv") / "input.csv"
    X[ALL_FEATURES].head(10).to_csv(csv_path, index=False)
    return csv_path


def test_predict_csv_happy_path(artifacts, input_csv) -> None:
    """predict_csv returns a DataFrame with 'xgb' and 'ensemble_mean' columns, 10 rows."""
    from ports_dfl.inference import predict_csv

    out = predict_csv(input_csv, artifacts)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 10
    assert set(out.columns) == {"xgb", "ensemble_mean"}


def test_predict_csv_predictions_are_finite_and_nonnegative(artifacts, input_csv) -> None:
    """All predictions must be finite numbers >= 0 (service time can't be negative)."""
    from ports_dfl.inference import predict_csv

    out = predict_csv(input_csv, artifacts)
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy())), f"Non-finite values in column {col!r}"
        assert (out[col] >= 0).all(), f"Negative predictions in column {col!r}"


def test_predict_csv_ensemble_mean_matches_model_columns(artifacts, input_csv) -> None:
    """ensemble_mean must equal the row-wise mean of all model columns."""
    from ports_dfl.inference import predict_csv

    out = predict_csv(input_csv, artifacts)
    model_cols = [c for c in out.columns if c != "ensemble_mean"]
    expected = out[model_cols].mean(axis=1)
    np.testing.assert_allclose(out["ensemble_mean"].to_numpy(), expected.to_numpy(), rtol=1e-6)


def test_predict_csv_missing_column_raises_value_error(artifacts, tmp_path_factory, Xy) -> None:
    """A CSV missing one required feature raises ValueError naming the missing column."""
    from ports_dfl.inference import predict_csv

    X, _ = Xy
    drop_col = ALL_FEATURES[0]  # drop the first feature column
    bad_csv = tmp_path_factory.mktemp("bad_csv") / "bad.csv"
    X[ALL_FEATURES].drop(columns=[drop_col]).head(10).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match=drop_col):
        predict_csv(bad_csv, artifacts)


def test_predict_csv_missing_multiple_columns_names_all(artifacts, tmp_path_factory, Xy) -> None:
    """When two columns are missing, the ValueError message names both."""
    from ports_dfl.inference import predict_csv

    X, _ = Xy
    drop_cols = ALL_FEATURES[:2]
    bad_csv = tmp_path_factory.mktemp("bad_csv2") / "bad2.csv"
    X[ALL_FEATURES].drop(columns=drop_cols).head(10).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        predict_csv(bad_csv, artifacts)

    err_msg = str(exc_info.value)
    for col in drop_cols:
        assert col in err_msg, f"Missing column {col!r} not mentioned in error: {err_msg!r}"
