"""BAP instance descriptor — dependency-light (numpy only).

This dataclass is intentionally kept free of any optimization-backend imports
(no Pyomo, no PyEPO), so instance builders and their unit tests can run in
environments that don't have the solver stack installed. ``discrete_bap.py``
re-exports ``BAPInstance`` from here, so existing imports
(``from bap_optim.discrete_bap import BAPInstance``) keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# frozen=True: the instance is shared/read by many parts of the solver and
# should never be accidentally modified.
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

    n_vessels: int
    n_berths: int
    arrivals: np.ndarray
    weights: np.ndarray
    big_m: float = 1000.0
    # Optional extensions; None default is safe (frozen dataclass can't mutate it).
    latest_start: np.ndarray | None = None
    berth_compat: np.ndarray | None = None
    service: np.ndarray | None = None
    # Single shared navigation channel: transit time (hours) each vessel needs to
    # ENTER (before berthing) and to EXIT (after service). No two transits may
    # overlap. ``None`` => no channel modelled (original berth-only behaviour).
    channel_time: float | None = None

    # --- small read helpers (no mutation; safe on a frozen dataclass) ------

    def compatible(self, i: int, b: int) -> bool:
        """Whether vessel i may berth at b. True for all (i,b) when no matrix."""
        # No matrix => homogeneous berths: any vessel may use any berth.
        if self.berth_compat is None:
            return True
        return bool(self.berth_compat[i, b])

    def compatible_berths(self, i: int) -> list[int]:
        """List of berths vessel i may use."""
        if self.berth_compat is None:
            return list(range(self.n_berths))
        return [b for b in range(self.n_berths) if bool(self.berth_compat[i, b])]

    def is_service(self, i: int) -> bool:
        """Whether vessel i is a priority 'service' (no-wait window) vessel."""
        # Short-circuit avoids indexing None when there is no service array.
        return self.service is not None and bool(self.service[i])

    def latest(self, i: int) -> float | None:
        """Latest start lᵢ for vessel i, or None if no finite window."""
        if self.latest_start is None:
            return None
        v = float(self.latest_start[i])
        # np.inf/np.nan encode "no window"; only return a real, finite value.
        return v if np.isfinite(v) else None
