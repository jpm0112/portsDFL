"""Sanity checks for the dataset loader and feature schema."""

# `import numpy as np` loads the NumPy library and gives it the short alias `np`.
import numpy as np

# Pull the feature-schema constants out of the project config module so the
# tests check the loader against the SAME single source of truth the code uses.
from ports_dfl.config import (
    ALL_FEATURES,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    NUMERIC_FEATURES,
    TARGET_COL,
)
# The functions under test, imported from the (still-to-be-written) data loader.
from ports_dfl.data.loader import (
    feature_role_summary,
    load_training_dataset,
    split_features_target,
)


# pytest AUTO-DISCOVERS any function whose name starts with `test_` and runs it.
# The `df` parameter is NOT a normal argument: it is the name of a pytest
# FIXTURE (defined in conftest.py). A fixture is reusable setup code; a test
# "requests" it simply by listing its name as a parameter, and pytest passes in
# whatever the fixture returns (here, the loaded DataFrame). `-> None` is a type
# hint saying the test returns nothing.
def test_load_returns_expected_shape(df) -> None:
    """Dataset has 5,589 rows × at least 18 columns (target + 17 features)."""
    # `assert EXPR` makes the test FAIL if EXPR is False. Here: row count is exact.
    # `df.shape` is (n_rows, n_cols), so `.shape[0]` is the number of rows.
    assert df.shape[0] == 5589
    # The thing we predict (the target column) must actually be present.
    assert TARGET_COL in df.columns


def test_no_missing_in_modelling_columns(df) -> None:
    """Modelling columns must be free of NAs (preprocessing should have handled them)."""
    # Build the list of columns models actually use: every feature plus the target.
    # `[TARGET_COL]` wraps the single string in a list so `+` can concatenate lists.
    cols = ALL_FEATURES + [TARGET_COL]
    # `.isna()` -> a True/False table of "is this cell missing?". The first
    # `.sum()` counts missing per column; the second sums those into one total.
    # We assert that grand total is 0, i.e. no missing values anywhere.
    assert df[cols].isna().sum().sum() == 0


def test_target_is_positive_and_varied(df) -> None:
    """Service time is strictly positive and has substantial variability."""
    target = df[TARGET_COL]
    # A service time of 0 or negative would be physically impossible -> every
    # value must be > 0. `.all()` is True only if EVERY element passes the test.
    assert (target > 0).all()
    # Guard against a degenerate/constant target: there should be lots of
    # distinct values. `.nunique()` counts the number of unique values.
    assert target.nunique() > 100
    # Sanity-check the scale of the target: a plausible mean (in hours), not a
    # tiny number near 0 nor an absurd one. (`<` chains: 1.0 < mean < 200.0.)
    assert 1.0 < target.mean() < 200.0


# No `df` here: this test only checks the config lists, so it needs no fixture.
def test_feature_roles_cover_all_features() -> None:
    """The three role lists partition ALL_FEATURES with no overlap or omissions."""
    roles = feature_role_summary()
    # `set(...)` turns a list into a set (unordered, no duplicates); `|` is set
    # UNION. Combining the three role lists should yield exactly ALL_FEATURES.
    union = set(roles["low_cardinality"]) | set(roles["high_cardinality"]) | set(roles["numeric"])
    assert union == set(ALL_FEATURES)
    # The union check alone would still pass if a feature appeared in TWO roles.
    # Comparing total lengths catches that double-counting: the three list
    # lengths must add up to the number of features exactly once each.
    assert (
        len(roles["low_cardinality"])
        + len(roles["high_cardinality"])
        + len(roles["numeric"])
        == len(ALL_FEATURES)
    )


def test_split_features_target_returns_aligned(Xy) -> None:
    # The `Xy` fixture returns a (features, target) pair; this line UNPACKS the
    # two returned objects into separate variables X and y in one step.
    X, y = Xy
    # Arrange/Act happened in the fixture; below is the ASSERT phase.
    # Features and target must have the same number of rows (aligned samples).
    assert len(X) == len(y)
    # The target must NOT leak into the feature matrix (would cause data leakage).
    assert TARGET_COL not in X.columns
    # X must contain exactly the configured feature columns — no more, no fewer.
    # `set(...)` comparison ignores column ORDER, checking membership only.
    assert set(X.columns) == set(ALL_FEATURES)


# `tmp_path` is a BUILT-IN pytest fixture: a fresh, unique temporary directory
# (a pathlib.Path) created for this test and cleaned up automatically afterward.
def test_load_with_explicit_path(tmp_path) -> None:
    """Loader respects an explicit path argument."""
    # ARRANGE: load the real dataset, then write its first 100 rows to a temp CSV.
    df = load_training_dataset()
    out = tmp_path / "tiny.csv"  # `/` joins paths on a pathlib.Path object
    df.head(100).to_csv(out, index=False)
    # ACT: load again, this time pointing the loader at our explicit small file.
    df_small = load_training_dataset(out)
    # ASSERT: it read the file we gave it (100 rows), not the default dataset.
    assert len(df_small) == 100


def test_role_lists_match_config_constants() -> None:
    """Sanity: feature_role_summary mirrors the config lists exactly."""
    roles = feature_role_summary()
    # Each role list must equal its config constant EXACTLY (order included),
    # so the summary helper can never drift out of sync with config.py.
    assert roles["low_cardinality"] == LOW_CARDINALITY_CATEGORICAL
    assert roles["high_cardinality"] == HIGH_CARDINALITY_CATEGORICAL
    assert roles["numeric"] == NUMERIC_FEATURES


def test_numeric_features_are_finite(df) -> None:
    """All numeric feature columns must be finite floats/ints."""
    # Loop over each numeric column and assert it has no NaN/inf values —
    # non-finite numbers would break math inside the models.
    for col in NUMERIC_FEATURES:
        # `np.isfinite(...)` is True for ordinary numbers, False for NaN/±inf.
        # The text after the comma is an assert MESSAGE shown only on failure;
        # the f-string (`f"... {col}"`) drops the offending column name into it.
        assert np.isfinite(df[col]).all(), f"Non-finite values found in {col}"
