# Code review findings

Findings from a repo-wide review done alongside adding beginner-friendly comments.
**Bugs fixed** were applied in place (look for `# FIX:` comments). **Issues reported** were
left unchanged (behavior-altering or judgment-dependent) — your call.

Severity: 🔴 likely bug / correctness · 🟡 robustness / edge case · ⚪ style / cleanup.

---

## `prediction_models/src/ports_dfl/optim/` (step A)

### Bugs fixed
- 🔴 **`berths.py` · `derive_berths_from_history`** — the loop iterated *stringified* index
  labels (`ct.index.astype(str)`) but looked rows up against the original index
  (`ct.loc[berth_name]`). With an **integer** berth column (e.g. `Sitio` site IDs, which the
  docstring explicitly supports) the string key `'1'` doesn't match integer index `1` →
  `KeyError`, crashing the data-driven path. **Fixed:** iterate the original index sorted by
  string form (`sorted(ct.index, key=lambda x: str(x))`), still emit string berth names.
- 🟡 **`weekly_instance.py` · `generate_synthetic_weekly_instance`** — if `n_services >
  n_vessels` (or negative), `is_service[chosen]` indexed out of bounds → `IndexError`.
  **Fixed:** clamp with `k = max(0, min(n_services, n_vessels))`.
- ⚪ **`optim/__init__.py` · `TYPE_CHECKING` block** — the three `schedule` exports
  (`assemble_schedule`, `compute_kpis`, `berth_index`) were in `_LAZY`/`__all__` but missing
  from the type-checking imports, so IDEs/type-checkers couldn't see them. **Fixed:** added
  the import (type-check-only; zero runtime change).

### Issues reported (not changed)
- 🔴 **`classic_bap.py` · `make_classic_problem` big-M** — `big_m = horizon + 4*tau_mean +
  arrivals.max()` can be **too small** for high-contention instances. The precedence
  constraint needs `M` to exceed the worst-case `s[i]+tau[i]-s[j]`, whose worst case is all
  `N` vessels stacked at one berth (≈ `arrivals.max() + sum(tau)`). With defaults that bound
  (~85) can be below the worst-case pile-up (~99), which would let a too-small `M` *spuriously
  force* precedence even when `z=0`, distorting schedules and the DFL regret signal. **Why not
  auto-fixed:** the correct bound needs a worst-case analysis and changes solver behavior, and
  the solver stack isn't runnable in this clone to validate. **Suggested:** `big_m =
  arrivals.max() + n_vessels * tau_mean * (1 + k·tau_sigma) + margin` (a provably-safe upper
  bound). *(Note: `generate_bap_instance` in `discrete_bap.py` uses the looser, safe
  `horizon + 400`.)*
- 🟡 **`discrete_bap.py` · `solve()` accepts `maxTimeLimit`** — if Gurobi hits the 60 s
  `TimeLimit` before the 0.5 % MIP gap, `solve()` accepts the incumbent silently, so the
  docstring's "within 0.5 % of optimum" guarantee (and clean regret comparisons) can be
  violated on hard instances. **Suggested:** on `maxTimeLimit`, check the achieved gap (accept
  only if ≤ 0.005) or log it; at minimum soften the docstring.
- 🟡 **`discrete_bap.py` · hard-window diagnostic** — if a service vessel has `latest(i) <
  arrivals[i]`, the start-var bounds become `(lo > hi)` and the solve fails with a generic
  "infeasible" rather than a clear "window earlier than arrival" message. **Suggested:** an
  upfront `ValueError` in `_getModel` for that case.
- 🟡 **`berths.py` · `derive_berths_from_history` `min_count=1`** — a single historical
  co-occurrence (possibly a data-entry error) makes a vessel type permanently compatible with
  a berth. Consider a higher default or a relative-frequency threshold.
- ⚪ **`discrete_bap.py` · dead import** — `from pyepo import EPO` is imported but never used.
  Left annotated; safe to delete (nothing imports `EPO` via this module).
- ⚪ **`weekly_instance.py`** — comment says services are "most-separated" but the code takes
  the *earliest* container calls (`container_idx[:k]`); also `VESSEL_TYPE_GROUPS` is imported
  but unused, and `week_end` recomputes the existing `we`. Cosmetic.
- ⚪ **`classic_bap.py`** — no validation that `contention > 0` / `tau_mean > 0`; `contention=0`
  raises `ZeroDivisionError`, negative values yield a nonsensical instance.

### Commenting
All 7 files (`instance.py`, `discrete_bap.py`, `berths.py`, `weekly_instance.py`,
`schedule.py`, `__init__.py`, `classic_bap.py`) got meaningful beginner-level comments
explaining Python syntax (decorators, dataclasses, type hints, comprehensions, lambdas,
`self`/`-> None`, numpy/pandas idioms, Pyomo objects) and the non-obvious logic.
