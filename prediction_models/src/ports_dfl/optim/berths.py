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

# `dataclass` is a decorator (see below) that auto-generates boilerplate
# methods (__init__, __repr__, __eq__) for a class that mainly holds data.
from dataclasses import dataclass

import numpy as np

# A tuple (round brackets) is an *immutable* sequence: once created, its
# contents can't be changed. Using a tuple here signals "this list of groups
# is a fixed constant". ALL-CAPS names are a Python convention for constants.
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


# `@dataclass` is a *decorator*: it wraps the class below and auto-writes the
# constructor and helper methods from the field declarations. `frozen=True`
# makes instances *immutable* (you can't reassign `.name` after creation) and,
# as a bonus, makes them hashable so a Berth can live in a set or dict key.
@dataclass(frozen=True)
class Berth:
    """A berth (or terminal) and the vessel-type groups it can serve.

    Fields:
        name:         human-readable berth/terminal identifier.
        served_types: frozenset of ``vessel_type_group`` values this berth
            can physically handle.
    """

    # These two lines are *type-annotated fields*, not assignments. The
    # dataclass decorator reads them and builds `__init__(self, name, served_types)`
    # for you. `name: str` means "name is a string"; the annotation is a hint,
    # Python does not enforce it at runtime.
    name: str
    # `frozenset[str]` is the immutable cousin of `set`: an unordered collection
    # of unique strings that can't be modified. It's used here because a frozen
    # dataclass needs hashable (immutable) fields, and a regular `set` is not.
    served_types: frozenset[str]

    # A *method*: a function defined inside a class. `self` is the instance the
    # method is called on (e.g. in `berth.can_serve("X")`, self is `berth`).
    # `-> bool` is the return-type hint: this returns True/False.
    def can_serve(self, vessel_type_group: str) -> bool:
        # `in` does a membership test; on a (frozen)set this is a fast O(1)
        # lookup, so checking compatibility is cheap.
        return vessel_type_group in self.served_types


# --- Documented best-guess catalog (validate against the port; see Q5) ------
#
# Terminal-level model of San Antonio. EPSA is merged into QC upstream
# (build_clean_dataset). These served-type sets are an informed default, NOT
# an authoritative berth specification — confirm with the port before relying
# on them operationally. Passenger / Other are routed to the multipurpose
# terminal as a permissive fallback so no vessel is left without a berth.
# `tuple[Berth, ...]` is a type hint meaning "a tuple of any number of Berth
# objects" (the `...` literally means "and so on"). Each `Berth(...)` call runs
# the auto-generated constructor. `{"Container"}` (curly braces) is a set
# literal; `frozenset({...})` freezes it into the immutable form the field wants.
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

    # `pd.crosstab` builds a contingency table: a 2-D count of how often each
    # pair of values co-occurs. Here rows = berths, columns = vessel types, and
    # each cell `ct.loc[berth, type]` = how many times that type was served at
    # that berth. `df[berth_col]` selects one column of the DataFrame.
    ct = pd.crosstab(df[berth_col], df[type_col])
    berths: list[Berth] = []  # empty list we'll append to; type hint is just a note
    # `ct.index` is the row labels (the distinct berth names). We sort so the
    # output order is stable/deterministic.
    # FIX: was `sorted(ct.index.astype(str))`, which sorts STRINGIFIED labels
    # but then `ct.loc[name]` looks them up against the ORIGINAL index. If the
    # berth column is non-string (e.g. integer Sitio IDs), the string key
    # mismatches the int index and raises KeyError. Iterate the original index
    # values (sorted by their string form) so the .loc lookup always matches.
    for berth_key in sorted(ct.index, key=lambda x: str(x)):
        # `ct.loc[berth_key]` selects that berth's row: a Series of per-type counts.
        row = ct.loc[berth_key]
        # A *set comprehension*: build a frozenset by looping over the row.
        # `row.items()` yields (column_label, count) pairs; we keep a type `t`
        # only when its count `n` meets the threshold. `int(n)` guards against
        # numpy integer types so the `>=` comparison behaves like plain ints.
        served = frozenset(str(t) for t, n in row.items() if int(n) >= min_count)
        # `str(berth_key)` normalises the stored name to a string regardless of
        # the source column's dtype.
        berths.append(Berth(str(berth_key), served))
    return berths


# The `|` in a type hint means "either type": `list[str] | np.ndarray` accepts
# a Python list OR a numpy array. `berths=DEFAULT_BERTHS` is a default value
# used when the caller omits it (safe here because the tuple is immutable —
# beware mutable defaults like `[]` or `{}`, which are shared between calls).
# The bare `*` marks everything after it as *keyword-only*: callers must write
# `allow_unmatched=True`, not pass it positionally.
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
    # A *list comprehension*: a compact `for` loop that builds a list. This
    # coerces every vessel-type entry to a string so comparisons are uniform.
    types = [str(t) for t in vessel_type_groups]
    # Tuple unpacking: assign two variables at once. N = number of vessels,
    # B = number of berths.
    N, B = len(types), len(berths)
    # Pre-allocate an (N, B) array filled with False; we'll flip cells to True.
    # `dtype=bool` makes it a boolean matrix (memory-efficient, and what the
    # downstream BAP model expects).
    compat = np.zeros((N, B), dtype=bool)
    # `enumerate` yields (index, value) pairs, so `i` is the vessel's row index
    # and `t` is its type string.
    for i, t in enumerate(types):
        # Build this vessel's compatibility row: one True/False per berth.
        row = np.array([berth.can_serve(t) for berth in berths], dtype=bool)
        # `.any()` is True if at least one entry is True; `not row.any()` means
        # this vessel's type matched NO berth in the catalog.
        if not row.any():
            if allow_unmatched:
                # `row[:] = True` is numpy slice-assignment: set every element
                # to True (route the vessel to all berths as a fallback).
                row[:] = True
            else:
                # f-string (the `f"..."` prefix) lets `{expr}` embed values.
                # `{t!r}` uses repr() so the type is shown quoted, e.g. 'Foo'.
                raise ValueError(
                    f"Vessel {i} of type {t!r} is served by no berth in the catalog "
                    f"({[b.name for b in berths]}). Pass allow_unmatched=True to "
                    f"route it to all berths, or extend the catalog."
                )
        # Write the finished row into row `i` of the matrix.
        compat[i] = row
    return compat


def berth_names(berths: list[Berth] | tuple[Berth, ...] = DEFAULT_BERTHS) -> list[str]:
    """Convenience: the ordered list of berth names (column order of compat)."""
    # List comprehension that pulls the `.name` out of each Berth, preserving
    # order so the names line up with the columns of the compat matrix above.
    return [b.name for b in berths]
