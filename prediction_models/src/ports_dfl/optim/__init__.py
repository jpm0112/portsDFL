"""Downstream optimization models for Decision-Focused Learning.

Exposes the discrete Berth Allocation Problem (DBAP) MILP, its decision-quality
utilities, the berth/compatibility catalog, and the week-long instance builder.

Imports are LAZY: pulling a solver-dependent name (``DiscreteBAP`` and the
helpers in ``discrete_bap``) imports PyEPO/Pyomo only on first access. The
dependency-light pieces — ``BAPInstance`` (numpy only), the berth catalog, and
the weekly-instance builder (numpy/pandas only) — import without the solver
stack, so instance building and its tests run in environments that lack PyEPO.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Dependency-light: safe to import eagerly (numpy only).
from ports_dfl.optim.instance import BAPInstance

# name -> submodule providing it (imported on first access via __getattr__).
_LAZY: dict[str, str] = {
    # Solver-dependent (PyEPO/Pyomo) — discrete_bap.
    "DiscreteBAP": "discrete_bap",
    "generate_bap_instance": "discrete_bap",
    "derive_starts_under_true_tau": "discrete_bap",
    "extract_decision": "discrete_bap",
    "schedule_cost_under_true_tau": "discrete_bap",
    # Dependency-light (numpy / pandas) — berths + weekly builder.
    "Berth": "berths",
    "DEFAULT_BERTHS": "berths",
    "VESSEL_TYPE_GROUPS": "berths",
    "derive_berths_from_history": "berths",
    "vessel_berth_compat": "berths",
    "berth_names": "berths",
    "build_weekly_instance": "weekly_instance",
    "generate_synthetic_weekly_instance": "weekly_instance",
    "WeeklyInstanceBundle": "weekly_instance",
    "assemble_schedule": "schedule",
    "compute_kpis": "schedule",
    "berth_index": "schedule",
}

__all__ = ["BAPInstance", *sorted(_LAZY)]


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(mod, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static-analysis hints only; not executed at runtime
    from ports_dfl.optim.berths import (  # noqa: F401
        DEFAULT_BERTHS,
        VESSEL_TYPE_GROUPS,
        Berth,
        berth_names,
        derive_berths_from_history,
        vessel_berth_compat,
    )
    from ports_dfl.optim.discrete_bap import (  # noqa: F401
        DiscreteBAP,
        derive_starts_under_true_tau,
        extract_decision,
        generate_bap_instance,
        schedule_cost_under_true_tau,
    )
    from ports_dfl.optim.weekly_instance import (  # noqa: F401
        WeeklyInstanceBundle,
        build_weekly_instance,
        generate_synthetic_weekly_instance,
    )
