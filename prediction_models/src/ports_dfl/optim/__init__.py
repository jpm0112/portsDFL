"""Downstream optimization models for Decision-Focused Learning.

Exposes the discrete Berth Allocation Problem (DBAP) MILP, its decision-quality
utilities, the berth/compatibility catalog, and the week-long instance builder.

Imports are LAZY: pulling a solver-dependent name (``DiscreteBAP`` and the
helpers in ``discrete_bap``) imports PyEPO/Pyomo only on first access. The
dependency-light pieces — ``BAPInstance`` (numpy only), the berth catalog, and
the weekly-instance builder (numpy/pandas only) — import without the solver
stack, so instance building and its tests run in environments that lack PyEPO.
"""

# `from __future__ import annotations` makes every type hint in this file be
# stored as plain text instead of being evaluated when the file loads. That is
# what lets us write hints like `dict[str, str]` and reference classes that are
# only imported lazily, without forcing those heavy imports to run at import time.
from __future__ import annotations

# `importlib` is the standard-library module that lets us import another module
# *by its name as a string* at runtime (see `import_module` below). We need that
# because we decide which submodule to load only when an attribute is requested.
import importlib

# `TYPE_CHECKING` is a special flag that is False at runtime but True when a type
# checker (mypy / your IDE) analyses the code. We use it at the bottom of the file
# to give static tools real imports without running them during normal execution.
from typing import TYPE_CHECKING

# Dependency-light: safe to import eagerly (numpy only).
# This is a normal, immediate import — `BAPInstance` is always available the
# moment this package is imported, because it does not drag in the solver stack.
from ports_dfl.optim.instance import BAPInstance

# A dictionary mapping each public name -> the submodule (filename without .py)
# that defines it. `dict[str, str]` is a type hint saying "keys are strings,
# values are strings". Nothing is imported here yet; this is just a lookup table
# that `__getattr__` (below) consults the first time someone accesses a name.
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

# `__all__` is the official list of public names for this package: it controls
# what `from ports_dfl.optim import *` exports, and tools use it as the public API.
# `"BAPInstance"` is listed explicitly (it is imported eagerly above), then the
# `*sorted(_LAZY)` part unpacks the alphabetically sorted dictionary keys into
# this list. (`*` here means "spread the items of this iterable into the list";
# iterating a dict yields its keys, so this adds every lazy name.)
__all__ = ["BAPInstance", *sorted(_LAZY)]


# PEP 562 lets a *module* define `__getattr__`, which Python calls automatically
# whenever someone accesses a module attribute that does not already exist.
# `def name(args) -> return_type:` defines a function; `name: str` is a parameter
# type hint (the requested attribute name, a string). This is the heart of the
# lazy-import trick: the heavy submodule is only loaded when its name is first used.
def __getattr__(name: str):  # PEP 562 lazy attribute loading
    # Look up which submodule owns this name. `.get(name)` returns None if the
    # name is not in the map (instead of raising), so we can handle it gracefully.
    submodule = _LAZY.get(name)
    if submodule is None:
        # Not one of our lazy names -> behave like a normal missing attribute.
        # `f"..."` is an f-string (text with embedded `{...}` expressions); `!r`
        # inserts the repr() form (quoted), e.g. produces "module 'x' has ...".
        # `__name__` is this module's dotted import path (e.g. "ports_dfl.optim").
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Import the owning submodule *now*, by building its full dotted path as a
    # string (e.g. "ports_dfl.optim.discrete_bap"). This is where PyEPO/Pyomo get
    # pulled in, but only if a solver-dependent name was actually requested.
    mod = importlib.import_module(f"{__name__}.{submodule}")
    # Pull the requested object (class/function) out of the freshly imported module.
    value = getattr(mod, name)
    # `globals()` is this module's namespace dict; storing the value there means
    # the attribute now "exists", so future accesses find it directly and never
    # re-enter __getattr__ (which is why the import work happens at most once).
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


# PEP 562 also lets a module define `__dir__`, which controls what `dir(module)`
# shows (and tab-completion in many tools). Without this, lazily-loaded names
# would not appear until after they were first accessed, so we report them all.
# `-> list[str]` is a return-type hint meaning "returns a list of strings".
def __dir__() -> list[str]:
    return sorted(__all__)


# This block runs ONLY during static type checking (`TYPE_CHECKING` is False at
# runtime), so these imports never execute when the program actually runs — they
# don't trigger the heavy solver imports. Their purpose is purely to tell type
# checkers and IDEs where each lazy name really comes from, enabling autocomplete
# and type inference. `# noqa: F401` silences the linter's "imported but unused"
# warning, since the names are intentionally imported only for the type checker.
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
    # FIX: the schedule submodule's lazy names (assemble_schedule, compute_kpis,
    # berth_index) are in _LAZY/__all__ but were missing from this block, so type
    # checkers/IDEs could not see them. Added for parity (type-check-only; no
    # runtime behavior change since TYPE_CHECKING is False at runtime).
    from ports_dfl.optim.schedule import (  # noqa: F401
        assemble_schedule,
        berth_index,
        compute_kpis,
    )
    from ports_dfl.optim.weekly_instance import (  # noqa: F401
        WeeklyInstanceBundle,
        build_weekly_instance,
        generate_synthetic_weekly_instance,
    )
