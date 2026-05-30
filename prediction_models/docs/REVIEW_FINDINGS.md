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

---

## `prediction_models/src/ports_dfl/` core — config/metrics/models/train/tuning (step B)

### Bugs fixed
- 🟡 **`train/pto.py` · `predict_pto`** — crashed on empty input (`np.concatenate([])` →
  "need at least one array to concatenate"). **Fixed:** early `return np.empty((0,), float32)`.

### Issues reported (not changed)
- 🟡 **`models/node.py` · `tree_depth` ignored** — accepted by `NODE`/`_NODERegressor` but never
  forwarded to `DenseODSTBlock`, so any sweep over `tree_depth` has no effect. Fix = pass
  `depth=tree_depth` to the block (verify the exact kwarg against the installed library; old
  checkpoints would change). Not auto-fixed (library absent, architecture-altering).
- 🟡 **`models/realmlp.py` · `depth` ignored unless `hidden_dim` set** — `hidden_sizes` is only
  built inside `if hidden_dim is not None`, so `RealMLP(depth=5)` alone does nothing.
- 🟡 **`models/tabm.py` & `node.py` · validation split** — uses the *unshuffled tail* of the
  training data; if rows are time/group-ordered, early stopping is miscalibrated. Tiny folds can
  also leave an empty train set. Shared design choice — fix consistently if at all.
- 🟡 **`models/log_target.py`** — docstring claims the back-transform is non-negative, but
  `predict` returns `exp(clip(·)) − offset`, which can be negative; and `load()` stores
  `inner_class`/`inner_module` metadata that it never uses to reconstruct the inner model (caller
  must pre-build a matching inner).
- 🟡 **`tuning/runner.py`** — a `MedianPruner` is configured but the objective never calls
  `trial.report()`/`should_prune()`, so pruning never triggers (dead config); an empty `splits`
  list yields `nan` (with a numpy warning) instead of a clear error.
- 🟡 **`train/dfl_blackbox.py` & `dfl_perturbed.py`** — `_evaluate_regret` does `np.mean(regrets)`
  on a possibly-empty list → `nan`, so early stopping never triggers and `best_val_regret` stays
  `inf`. (Also documented the intended asymmetry: the training loss uses the solver's raw `starts`
  while regret re-derives feasible starts — this is the standard PyEPO pattern, left as-is.)
- ⚪ **`metrics/regression.py`** — `all_metrics` isn't in `metrics/__init__.py`'s `__all__` (so
  `from ports_dfl.metrics import all_metrics` fails; the codebase imports it from
  `.regression` directly, so nothing breaks today); `summarize_folds` uses `ddof=1` → `NaN` for a
  single fold; `mae/rmse/r2` skip the `np.asarray(float)` coercion that `mape` applies.
- ⚪ **`train/pto.py`** — cosine `T_max=max_epochs` means the LR never fully anneals when early
  stopping fires; `best_epoch` is 0-indexed while `epochs_run` is a count.
- ⚪ **`models/baselines.py`** — `GroupMeanBaseline.fit` would overwrite a pre-existing `_target`
  column (extremely unlikely for this internal API).

### Commenting
All config/metrics/models/train/tuning files annotated: ABCs & `@abstractmethod`,
`nn.Module`/`forward`/`super().__init__()`, decorators, dataclasses & `field(default_factory)`,
type hints (`| None`, `Literal`, forward-ref returns), torch `save`/`load`/`state_dict`, AMP
GradScaler flow, Optuna `suggest_*`, and numpy/pandas idioms. (Note: `tabm.py` had a transient
accidental edit during annotation that the agent repaired — verified intact + compiles.)

---

## `prediction_models/scripts/` (step C)

### Bugs fixed
- 🟡 **`plan_week.py` · `_write_csv`** — `rows[0].keys()` with no guard → `IndexError` if the
  solver schedules zero vessels. **Fixed:** early `return` on empty `rows`.
- 🟡 **`run_dfl_synthetic.py`** — the predictive-summary CSV hard-coded the DFL row label
  `"DFL (blackbox)"`, mislabeling runs done with `--method perturbed`. **Fixed:** use the
  computed `dfl_tag`.
- ⚪ **`run_dfl_real_bap.py`** — `_evaluate_decisions` was hinted `-> dict` but returns a
  `(summary, df)` tuple. **Fixed:** corrected to `-> tuple[dict, pd.DataFrame]` (annotation only).
- ⚪ **`benchmark_dbb.py`** — removed unused `import os`.

### Issues reported (not changed)
- 🟡 **`build_report.py`** — `ax.boxplot(..., labels=[...])` uses the `labels=` kwarg deprecated
  in matplotlib 3.9 and **removed in 3.11** → `TypeError` on newer matplotlib. Switch to
  `tick_labels=`. Also: CDF/boxplot pages divide by `len(...)` without guarding an empty CSV;
  `.iloc[0]` after a `str.contains("PtO"/"DFL")` filter can `IndexError` if labels are renamed.
- 🟡 **`compare.py`** — display label uses `fname.replace('cv_summary','').strip('_.csv')`;
  `str.strip` strips any of the *characters* `_.csv`, so `cv_summary_stock.csv` → `"tock"`
  (drops the leading `s`). Use `Path(fname).stem.replace('cv_summary','').strip('_')`. Cosmetic
  (label only). Also `_read_summary` can `KeyError` on a malformed summary CSV.
- 🟡 **`run_realmlp.py`** — the tuned re-evaluation `RealMLP(input_dim=..., **best_params)` does
  **not** pass `n_epochs`, so it uses RealMLP's class default instead of `--n_epochs` (the stock
  and tuning runs do pass it). Inconsistent epochs between tuning and final eval.
- ⚪ **`run_linear.py`** — `Ridge(random_state=SEED)` is a no-op under the default closed-form
  solver (only `sag`/`saga` use it); the sklearn-vs-PyTorch `alpha = weight_decay·n` mapping is
  approximate, so the "sanity-check" MAE may legitimately diverge.
- ⚪ **`run_tabm.py` / `run_node.py`** — sharing a `--study_name` resumes the existing Optuna
  SQLite study (`load_if_exists=True`), so one model could append to another's study; and
  `_evaluate_best` hardcodes `categorical_strategy="target"` (would mismatch if that's ever
  tuned). Safe with current defaults.
- ⚪ **`benchmark_dbb.py`** — stale docstring ("median of 10 solves" / "1 epoch" vs the code's 12
  solves-minus-cold-start / `max_epochs=5`); header/data column widths don't align.
- 🟡 **Several scripts** — relative-percent prints (`run_dfl_real_bap.py`,
  `run_dfl_synthetic.py`) divide by the FI/PtO mean without a zero guard → `inf`/`nan` if that
  baseline is exactly 0 (possible on very-low-contention synthetic instances).

### Commenting
All 13 scripts annotated: argparse (flags, `store_true`, mutually-exclusive groups), `sys.path`
bootstrapping, `if __name__ == "__main__":`, the PtO/DFL pipeline steps, `@contextmanager`
timing, and pandas/numpy/matplotlib/reportlab idioms.
