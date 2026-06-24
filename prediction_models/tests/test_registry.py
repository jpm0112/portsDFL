"""Tests for the model registry — ports_dfl.models.registry.

Pins the public contract of MODELS and get_spec(): the exact set of 7 registered
model names, their correct cls/kind/seed_kwarg/early_stopping values, and the
KeyError behaviour on unknown names.  Tests assert the spec *contents*, not just
that the dict is non-empty, so a mis-configured entry will still be caught.
"""

import dataclasses

import pytest

from ports_dfl.models.registry import MODELS, ModelSpec, get_spec

# ---------------------------------------------------------------------------
# MODELS dict — structural checks
# ---------------------------------------------------------------------------

EXPECTED_NAMES = {"xgb", "lgbm", "rf", "linear", "realmlp", "tabm", "node"}


def test_models_has_exactly_seven_entries() -> None:
    """MODELS must contain exactly the 7 deployable models — no more, no less."""
    assert set(MODELS.keys()) == EXPECTED_NAMES


def test_all_values_are_model_specs() -> None:
    """Every value in MODELS is a frozen ModelSpec dataclass instance."""
    for name, spec in MODELS.items():
        assert isinstance(spec, ModelSpec), f"{name!r} maps to {type(spec)!r}"
    # frozen: a mutation attempt must raise
    spec = MODELS["xgb"]
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        spec.kind = "tree"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_spec — happy path spot-checks
# ---------------------------------------------------------------------------

def test_get_spec_xgb_class() -> None:
    """get_spec('xgb').cls must be XGBoostRegressorModel."""
    from ports_dfl.models.xgb import XGBoostRegressorModel

    assert get_spec("xgb").cls is XGBoostRegressorModel


def test_get_spec_linear_class() -> None:
    """get_spec('linear').cls must be LinearRegressor."""
    from ports_dfl.models.linear import LinearRegressor

    assert get_spec("linear").cls is LinearRegressor


def test_get_spec_returns_same_object_as_models_dict() -> None:
    """get_spec is just a lookup — same object as MODELS[name]."""
    for name in EXPECTED_NAMES:
        assert get_spec(name) is MODELS[name]


# ---------------------------------------------------------------------------
# kind field: tree vs neural
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["xgb", "lgbm", "rf"])
def test_tree_models_have_kind_tree(name: str) -> None:
    assert get_spec(name).kind == "tree", f"{name!r} should be kind='tree'"


@pytest.mark.parametrize("name", ["linear", "realmlp", "tabm", "node"])
def test_neural_models_have_kind_neural(name: str) -> None:
    assert get_spec(name).kind == "neural", f"{name!r} should be kind='neural'"


# ---------------------------------------------------------------------------
# early_stopping field
# ---------------------------------------------------------------------------

def test_rf_early_stopping_is_false() -> None:
    """RandomForest has no boosting rounds — early_stopping must be False."""
    assert get_spec("rf").early_stopping is False


@pytest.mark.parametrize("name", ["xgb", "lgbm", "linear", "realmlp", "tabm", "node"])
def test_non_rf_models_have_early_stopping_true(name: str) -> None:
    assert get_spec(name).early_stopping is True, f"{name!r} should have early_stopping=True"


# ---------------------------------------------------------------------------
# seed_kwarg field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["xgb", "lgbm", "rf", "realmlp"])
def test_seed_kwarg_is_random_state_for_sklearn_style(name: str) -> None:
    assert get_spec(name).seed_kwarg == "random_state", (
        f"{name!r} seed_kwarg should be 'random_state'"
    )


@pytest.mark.parametrize("name", ["linear", "tabm", "node"])
def test_seed_kwarg_is_seed_for_pytorch_style(name: str) -> None:
    assert get_spec(name).seed_kwarg == "seed", (
        f"{name!r} seed_kwarg should be 'seed'"
    )


# ---------------------------------------------------------------------------
# suggest_fn field: must be callable
# ---------------------------------------------------------------------------

def test_all_suggest_fns_are_callable() -> None:
    for name, spec in MODELS.items():
        assert callable(spec.suggest_fn), f"{name!r}: suggest_fn must be callable"


# ---------------------------------------------------------------------------
# get_spec — error path
# ---------------------------------------------------------------------------

def test_get_spec_unknown_name_raises_key_error() -> None:
    """get_spec on an unknown name must raise KeyError (not return None, crash, etc.)."""
    with pytest.raises(KeyError):
        get_spec("does_not_exist")


def test_get_spec_error_message_contains_name() -> None:
    """The KeyError message should name the bad key so callers know what went wrong."""
    with pytest.raises(KeyError, match="unknown_model"):
        get_spec("unknown_model")
