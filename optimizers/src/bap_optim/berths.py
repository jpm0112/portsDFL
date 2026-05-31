"""Berth catalog and vessel–berth compatibility for the Port of San Antonio.

The discrete BAP (``discrete_bap.py``) accepts a ``berth_compat`` matrix of
shape ``(n_vessels, n_berths)`` of bool: ``True`` where a vessel may use a
berth. This module builds that matrix two ways:

1. ``derive_berths_from_history`` — the PREFERRED, data-driven construction:
   read the historical ``(berth × vessel_type_group)`` co-occurrence from the
   cleaned dataset and treat a vessel type as compatible with a berth iff it
   was historically served there above a small count threshold.
2. ``DEFAULT_BERTHS`` — a documented BEST-GUESS catalog used when the
   proprietary data is not available (e.g. the public repo clone). It is a
   starting point only and **must be validated against the port** (see the
   decisions log, Q5).

The "berth" granularity is configurable: pass ``berth_col="Terminal"`` for a
coarse 4–5 terminal model, or ``berth_col="Sitio"`` for the finer 9-site
model. ``vessel_type_group`` (a.k.a. ``Tipo nave (agrupado)`` in the
source-faithful dataset) is the 7-class vessel grouping defined in
``src/build_clean_dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 7 operational vessel-type groups (see src/build_clean_dataset.py).
VESSEL_TYPE_GROUPS = (
    "Container",
    "Dry Bulk",
    "Vehicle Carrier",
    "Liquid Bulk",
    "General Cargo",
    "Passenger",
    "Other",
)


# frozen=True makes Berth hashable (frozenset field), so it can live in a set
# or dict key.
@dataclass(frozen=True)
class Berth:
    """A berth (or terminal) and the vessel-type groups it can serve.

    Fields:
        name:         human-readable berth/terminal identifier.
        served_types: frozenset of ``vessel_type_group`` values this berth
            can physically handle.
    """

    name: str
    # frozenset (not set) so the frozen dataclass field stays hashable.
    served_types: frozenset[str]

    def can_serve(self, vessel_type_group: str) -> bool:
        return vessel_type_group in self.served_types


# --- Documented best-guess catalog (validate against the port; see Q5) ------
#
# Terminal-level model of San Antonio. EPSA is merged into QC upstream
# (build_clean_dataset). These served-type sets are an informed default, NOT
# an authoritative berth specification — confirm with the port before relying
# on them operationally. Passenger / Other are routed to the multipurpose
# terminal as a permissive fallback so no vessel is left without a berth.
DEFAULT_BERTHS: tuple[Berth, ...] = (
    Berth("STI", frozenset({"Container"})),
    Berth(
        "DP World",
        frozenset({"Container", "Vehicle Carrier", "General Cargo", "Passenger", "Other"}),
    ),
    Berth("PANUL", frozenset({"Dry Bulk"})),
    Berth("QC", frozenset({"Liquid Bulk"})),
)


def derive_berths_from_history(
    df,
    berth_col: str = "Terminal",
    type_col: str = "vessel_type_group",
    min_count: int = 1,
) -> list[Berth]:
    """Derive the berth catalog + served-type sets from historical calls.

    A vessel type is considered compatible with a berth iff it was served
    there at least ``min_count`` times in ``df``. This is the data-driven
    construction described in the decisions log (Q5). Raise ``min_count`` to
    drop rare one-off historical assignments (likely exceptions, not policy).

    Args:
        df: a DataFrame with the berth and vessel-type columns. Use the
            source-faithful ``clean_dataset.csv`` (columns ``Terminal`` /
            ``Sitio`` and ``Tipo nave (agrupado)``) or the engineered
            ``training_dataset.csv`` (``Terminal`` / ``vessel_type_group``).
        berth_col: column holding the berth/terminal identifier.
        type_col:  column holding the vessel-type group.
        min_count: minimum historical co-occurrences to count as compatible.

    Returns:
        A list of ``Berth`` (one per distinct berth value), ordered by berth
        name, each with the set of vessel-type groups it historically served.
    """
    import pandas as pd  # local import: keep module importable without pandas

    # Contingency table: rows = berths, columns = vessel types, cell = count
    # of how many times that type was served at that berth.
    ct = pd.crosstab(df[berth_col], df[type_col])
    berths: list[Berth] = []
    # FIX: was `sorted(ct.index.astype(str))`, which sorts STRINGIFIED labels
    # but then `ct.loc[name]` looks them up against the ORIGINAL index. If the
    # berth column is non-string (e.g. integer Sitio IDs), the string key
    # mismatches the int index and raises KeyError. Iterate the original index
    # values (sorted by their string form) so the .loc lookup always matches.
    for berth_key in sorted(ct.index, key=lambda x: str(x)):
        row = ct.loc[berth_key]
        # int(n) guards against numpy integer types in the >= comparison.
        served = frozenset(str(t) for t, n in row.items() if int(n) >= min_count)
        berths.append(Berth(str(berth_key), served))
    return berths


def vessel_berth_compat(
    vessel_type_groups: list[str] | np.ndarray,
    berths: list[Berth] | tuple[Berth, ...] = DEFAULT_BERTHS,
    *,
    allow_unmatched: bool = False,
) -> np.ndarray:
    """Build the ``(n_vessels, n_berths)`` boolean compatibility matrix.

    ``compat[i, b]`` is ``True`` iff ``berths[b]`` can serve vessel i's type.

    Args:
        vessel_type_groups: length-N sequence of each vessel's
            ``vessel_type_group``.
        berths: the berth catalog (defaults to ``DEFAULT_BERTHS``).
        allow_unmatched: if a vessel's type is served by NO berth, raising is
            the default (so unmodeled types surface loudly). Set ``True`` to
            instead mark such a vessel compatible with every berth (a
            permissive fallback that avoids structural infeasibility).

    Returns:
        ndarray (N, B) of bool.
    """
    types = [str(t) for t in vessel_type_groups]
    N, B = len(types), len(berths)
    compat = np.zeros((N, B), dtype=bool)
    for i, t in enumerate(types):
        row = np.array([berth.can_serve(t) for berth in berths], dtype=bool)
        # This vessel's type matched NO berth in the catalog.
        if not row.any():
            if allow_unmatched:
                row[:] = True  # permissive fallback: route to all berths
            else:
                raise ValueError(
                    f"Vessel {i} of type {t!r} is served by no berth in the catalog "
                    f"({[b.name for b in berths]}). Pass allow_unmatched=True to "
                    f"route it to all berths, or extend the catalog."
                )
        compat[i] = row
    return compat


def berth_names(berths: list[Berth] | tuple[Berth, ...] = DEFAULT_BERTHS) -> list[str]:
    """Convenience: the ordered list of berth names (column order of compat)."""
    return [b.name for b in berths]
