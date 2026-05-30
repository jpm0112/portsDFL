"""BAP instance descriptor — dependency-light (numpy only).

This dataclass is intentionally kept free of any optimization-backend imports
(no Pyomo, no PyEPO), so instance builders and their unit tests can run in
environments that don't have the solver stack installed. ``discrete_bap.py``
re-exports ``BAPInstance`` from here, so existing imports
(``from bap_optim.discrete_bap import BAPInstance``) keep working.
"""

# `from __future__ import annotations` makes Python treat all type hints below
# as plain text (not evaluated at runtime). This is what lets us write modern
# hints like `np.ndarray | None` even on older Python versions.
from __future__ import annotations

# `dataclass` is a decorator (see below) that auto-writes boilerplate for a
# class whose main job is to hold data — e.g. __init__, __repr__, ==.
from dataclasses import dataclass

import numpy as np


# A decorator (the `@name` line above a class/function) wraps that class to add
# behaviour. `@dataclass` reads the field declarations below and generates an
# __init__ constructor for us. `frozen=True` makes instances IMMUTABLE: once an
# object is created you cannot reassign its fields (doing so raises an error).
# Immutability is handy here because the instance is shared/read by many parts
# of the solver and should never be accidentally modified.
@dataclass(frozen=True)
class BAPInstance:
    """A DBAP instance.

    Fields:
        n_vessels: number of vessels.
        n_berths:  number of berths.
        arrivals:  ndarray (n_vessels,) — vessel arrival times in hours.
        weights:   ndarray (n_vessels,) — priority weights (≥0).
        big_m:     scalar M for precedence constraints. Should be ≥ horizon
            length plus the largest plausible total service time at one berth.

    Optional extensions (all default ``None`` for backward compatibility;
    the existing synthetic generators and trainers ignore them):
        latest_start: ndarray (n_vessels,) — latest start time lᵢ for each
            vessel. Non-finite (``np.inf``/``np.nan``) means "no window".
            Only consulted for vessels flagged in ``service``.
        berth_compat: ndarray (n_vessels, n_berths) of bool — ``True`` where
            vessel i may use berth b. ``None`` means every vessel may use
            every berth (homogeneous berths, original behaviour).
        service:      ndarray (n_vessels,) of bool — ``True`` for "service"
            (priority, no-wait) vessels that carry a hard/soft latest-start
            window. ``None`` means no service vessels.
    """

    # Each line below is a "field": `name: type` declares a piece of data the
    # object holds, and `type` is a hint (annotation) for humans/tools — Python
    # does not enforce it at runtime. `@dataclass` turns these into the
    # constructor's parameters, in this same order.
    n_vessels: int
    n_berths: int
    arrivals: np.ndarray
    weights: np.ndarray
    # A field with `= value` gives a DEFAULT, so the caller may omit it.
    # Note: dataclass fields with defaults must come AFTER fields without
    # defaults (that is why these are listed last).
    big_m: float = 1000.0
    # `np.ndarray | None` means "either a numpy array OR the value None".
    # Defaulting to None marks these as optional extensions. Using None (not a
    # real array) as the default is safe here: frozen dataclasses can't mutate
    # it, so there is no shared-mutable-default trap.
    latest_start: np.ndarray | None = None
    berth_compat: np.ndarray | None = None
    service: np.ndarray | None = None

    # --- small read helpers (no mutation; safe on a frozen dataclass) ------

    # `def name(self, ...) -> ReturnType:` defines a method (a function attached
    # to the object). `self` is the object itself — Python passes it in
    # automatically when you call `instance.compatible(...)`. `-> bool` is a
    # hint saying this returns a True/False value.
    def compatible(self, i: int, b: int) -> bool:
        """Whether vessel i may berth at b. True for all (i,b) when no matrix."""
        # No compatibility matrix supplied => homogeneous berths: any vessel
        # may use any berth, so always allow.
        if self.berth_compat is None:
            return True
        # `berth_compat[i, b]` indexes a 2D numpy array at row i, column b.
        # `bool(...)` converts numpy's bool (np.bool_) into a plain Python bool.
        return bool(self.berth_compat[i, b])

    def compatible_berths(self, i: int) -> list[int]:
        """List of berths vessel i may use."""
        # No matrix => every berth index 0..n_berths-1 is allowed.
        # `range(n)` yields 0,1,...,n-1; `list(...)` materialises it as a list.
        if self.berth_compat is None:
            return list(range(self.n_berths))
        # A list comprehension: build a list of berth indices `b` but keep only
        # those where the compatibility flag for (vessel i, berth b) is True.
        return [b for b in range(self.n_berths) if bool(self.berth_compat[i, b])]

    def is_service(self, i: int) -> bool:
        """Whether vessel i is a priority 'service' (no-wait window) vessel."""
        # Short-circuit: if there is no service array, the first operand is
        # False and the `self.service[i]` lookup is skipped (avoids indexing
        # None). Otherwise return whether vessel i's flag is set.
        return self.service is not None and bool(self.service[i])

    def latest(self, i: int) -> float | None:
        """Latest start lᵢ for vessel i, or None if no finite window."""
        # No latest-start array at all => this vessel has no window.
        if self.latest_start is None:
            return None
        # Read the i-th latest start and coerce to a plain Python float.
        v = float(self.latest_start[i])
        # `np.inf`/`np.nan` encode "no window"; only return a real, finite
        # value. `np.isfinite` is False for both infinity and NaN.
        return v if np.isfinite(v) else None
