"""
Tests for src.data_prep.

Covers:
    - time_split partitions are disjoint and exhaustive.
    - encode_categoricals produces dense 0..k-1 indices fit only on train.
    - apply_encoding maps unseen categories to the OOV sentinel.
    - add_log_target applies an exact natural logarithm.
    - prepare end-to-end yields train/test with all expected columns and
      no train rows containing OOV indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    OOV_INDEX,
    add_log_target,
    apply_encoding,
    encode_categoricals,
    prepare,
    time_split,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """
    Tiny synthetic dataset that mimics the BHM CSV structure.

    Output: DataFrame with 6 rows spanning 2020-2025, three vessel types,
            two berths, and two services. Used by all tests in this module.
    Description: Designed so that the 2025 row contains a service ('NEW')
                 unseen in 2020-2024 to exercise OOV handling.
    """
    return pd.DataFrame(
        {
            "Tipo nave (agrupado)": ["A", "B", "A", "C", "A", "B"],
            "Sitio": ["S1", "S2", "S1", "S2", "S1", "S2"],
            "Servicio": ["X", "Y", "X", "X", "Y", "NEW"],
            "service_time_hours": [10.0, 20.0, 15.0, 30.0, 12.0, 18.0],
            "atraque_year": [2020, 2021, 2022, 2023, 2024, 2025],
        }
    )


def test_time_split_is_disjoint_and_exhaustive(toy_df):
    """Train and test together cover every row exactly once."""
    train, test = time_split(toy_df, train_year_max=2024)
    assert len(train) + len(test) == len(toy_df)
    assert train["atraque_year"].max() <= 2024
    assert test["atraque_year"].min() >= 2025


def test_encode_categoricals_dense_zero_indexed(toy_df):
    """Each map covers exactly the train levels, indexed 0..k-1, no gaps."""
    train, _ = time_split(toy_df, train_year_max=2024)
    enc = encode_categoricals(train)
    assert sorted(enc.vessel.values()) == list(range(enc.n_vessel))
    assert sorted(enc.berth.values()) == list(range(enc.n_berth))
    assert sorted(enc.service.values()) == list(range(enc.n_service))
    # No leakage: 'NEW' service is in 2025 only and must not appear here.
    assert "NEW" not in enc.service


def test_apply_encoding_handles_oov(toy_df):
    """Test rows with unseen categories are mapped to OOV_INDEX."""
    train, test = time_split(toy_df, train_year_max=2024)
    enc = encode_categoricals(train)
    test_enc = apply_encoding(test, enc)
    # Test row has Servicio = 'NEW' which was not in training.
    assert (test_enc["service_idx"] == OOV_INDEX).all()


def test_apply_encoding_train_has_no_oov(toy_df):
    """Train rows must always map to non-OOV indices after encoding."""
    train, _ = time_split(toy_df, train_year_max=2024)
    enc = encode_categoricals(train)
    train_enc = apply_encoding(train, enc)
    assert (train_enc["vessel_idx"] != OOV_INDEX).all()
    assert (train_enc["berth_idx"] != OOV_INDEX).all()
    assert (train_enc["service_idx"] != OOV_INDEX).all()


def test_add_log_target_exact():
    """log_service_time equals np.log of the raw column to machine precision."""
    df = pd.DataFrame({"service_time_hours": [1.0, np.e, 100.0]})
    out = add_log_target(df)
    np.testing.assert_allclose(out["log_service_time"].to_numpy(), [0.0, 1.0, np.log(100.0)])


def test_prepare_end_to_end(tmp_path, toy_df):
    """prepare() returns train/test/Encoding/scaler with all derived columns present."""
    # Toy dataset needs the columns covariate computation expects.
    df = toy_df.copy()
    df["TRG"] = [10000, 20000, 15000, 30000, 12000, 18000]
    df["Calado diff"] = [0.5, -0.3, 1.0, -0.8, 0.0, 0.2]
    df["atraque_hour"] = [3, 14, 9, 20, 6, 22]
    df["atraque_dayofweek"] = [0, 1, 2, 3, 4, 5]

    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)
    train, test, enc, scaler = prepare(csv_path, train_year_max=2024)

    expected_cols = {"vessel_idx", "berth_idx", "service_idx", "log_service_time",
                     "z_log_trg", "z_abs_calado_diff", "z_hour_sin"}
    assert expected_cols.issubset(train.columns)
    assert expected_cols.issubset(test.columns)
    assert enc.n_vessel == train["Tipo nave (agrupado)"].nunique()
    assert enc.n_berth == train["Sitio"].nunique()
    assert enc.n_service == train["Servicio"].nunique()
    assert scaler is not None and "log_trg" in scaler.means


def test_prepare_without_covariates(tmp_path, toy_df):
    """prepare(with_covariates=False) skips the covariate columns and returns scaler=None."""
    csv_path = tmp_path / "toy.csv"
    toy_df.to_csv(csv_path, index=False)
    train, test, enc, scaler = prepare(csv_path, train_year_max=2024, with_covariates=False)
    assert scaler is None
    assert "z_log_trg" not in train.columns
