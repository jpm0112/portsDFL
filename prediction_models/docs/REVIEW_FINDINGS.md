# Code review findings

Findings from a repo-wide review done alongside adding beginner-friendly comments.
**Bugs fixed** were applied in place (look for `# FIX:` comments). **Issues reported** were
left unchanged (behavior-altering or judgment-dependent) — your call.

Severity: 🔴 likely bug / correctness · 🟡 robustness / edge case · ⚪ style / cleanup.

## Round 2 fixes (2026-07-18)

A second review pass (methodology + correctness). Fixes applied on branch
`review-fixes`; two items are **documented only** (your call on how to resolve).

**Applied:**
1. ✅ 🔴 **Objective/loss/metric mismatch (the load-bearing one).** The MILP's default
   objective is unweighted waiting `Σᵢ(sᵢ − aᵢ)`, but the DFL loss, the regret metric,
   the README and the report scored *weighted completion* `Σᵢ wᵢ(sᵢ + τᵢ)` — an objective
   the optimizer never minimises. Because the full-information benchmark minimises the
   unweighted objective, a prediction could beat it on the weighted metric → **regret was
   not guaranteed ≥ 0** (the `-1e-1` test tolerances were absorbing this, not float noise).
   Also, weights never entered the optimizer, so DFL could not learn to prioritise heavy
   vessels *through* it — the mechanism `run_dfl_synthetic` claims to show. **Fix:** dropped
   weights from `schedule_cost_under_true_tau` and both trainers' loss + regret eval; the
   cost is now `Σᵢ(sᵢ + τᵢ)`, which equals the objective up to a constant, so regret ≥ 0
   holds. Tightened the regret/FI-optimality test tolerances to `1e-3`; renamed the
   `weighted_cost_*` summary keys to `cost_*` across the run scripts, `compare.py`,
   `build_report.py`; corrected the README + report to state τ enters only the precedence
   constraints. *(Chosen direction: align to the unweighted objective. Putting weighted
   completion into the MILP objective — so weights drive scheduling and PyEPO's linear-cost
   assumption holds — is the alternative, but it invalidates the committed results and needs
   a full re-run.)*
2. ✅ 🔴 **`generate_bap_instance` big-M** — was `horizon + 400`, which does not scale with N
   and can silently cut valid schedules once a berth's cumulative completion exceeds it
   (this supersedes the note below at §step-A that called `horizon + 400` "looser, safe" —
   it is not safe). Now `max(arrivals) + N·service_time_ub`.
3. ✅ 🟡 **DFL `_evaluate_regret` empty-val `nan`** — `np.mean([])` silently disabled early
   stopping; now raises on an empty validation set.
4. ✅ 🟡 **`MIPGap=0` is Gurobi-only** — applied only for gurobi; other solvers now warn that
   the exact-optimality (hence regret ≥ 0) assumption is not enforced.
5. ✅ ⚪ Documented that `derive_starts_under_true_tau` is channel-agnostic; removed the dead
   `from pyepo import EPO`; guarded `test_discrete_bap` with `pytest.importorskip`;
   `summarize_folds` single-fold std no longer `nan`; guarded the `regret_relative_pct`
   zero-division. Added a seeded smoke regression test for the blackbox DFL loop
   (`tests/test_dfl_train.py`).

**Documented only (not changed — your decision):**
- 🔴 **`Calado diff` is suspected target leakage.** It is the arrival−departure draft, known
   only *after* service (`docs/project_description.md` flags draft difference as
   post-berthing), yet it is a training feature in `config.ALL_FEATURES`. `predictor/features.py`
   defaults it to 0 at inference and admits `covid_era` cutoffs are reverse-engineered —
   consistent with the real `data/training_dataset.csv` having been built with the post-hoc
   value. **Verify how that CSV computed `Calado diff` with the data owner; if post-hoc,
   remove it and re-tune.**
- 🟡 **`data/training_dataset.csv` is not reproducible.** The committed `data_pipeline`
   builders emit a different schema (`estadia_sitio_hours`, …) than the models consume
   (`service_time_hours`, `covid_era`, cyclical `atraque_*`, `Calado diff`). No committed
   script regenerates the consumed CSV, so the leakage-safe logic in `build_training_dataset.py`
   applies to a schema the models don't use. **Commit the actual feature-engineering step.**
- 🟡 **"Real BAP" uses synthetic scheduling geometry.** `run_dfl_real_bap.py` groups unrelated
   real vessels into instances by random permutation and gives them one fabricated
   arrival/weight vector from `generate_bap_instance`. τ is real; the "week" is not. State this
   as a limitation rather than implying real scheduling.

## Summary

~61 Python files across `prediction_models/` and repo-root `src/` were
reviewed and given beginner-friendly comments (in 5 steps, each its own commit). The repo
byte-compiles cleanly and the 17 BAP tests still pass against Gurobi after annotation.

**8 clear fixes applied** (all crash-safe / behavior-neutral):
`berths.py` integer-`Sitio` `KeyError`; `weekly_instance` `n_services` `IndexError`;
`optim/__init__` type-check imports; `pto.predict_pto` empty-input crash; `plan_week._write_csv`
empty-rows crash; `run_dfl_synthetic` DFL row label by method; `run_dfl_real_bap` return-type hint;
`benchmark_dbb` dead `import os`.

**Top issues — FIXED (2026-05-31):**
1. ✅ 🔴 **`build_training_dataset.py` group-feature leakage** — FIXED. The cross-vessel group
   averages (`type_terminal_avg_stay`, `type_avg_stay`, `terminal_avg_stay`) now use a
   completion-aware `_causal_group_mean`: each row averages only same-group visits whose
   `last_unmooring <= this row's first_mooring` (outcome already known at decision time). The
   per-vessel features were left as-is — a single physical vessel can't overlap its own prior
   visits, so they were never leaky. *Verified on a hand-built example (a naive expanding-shift
   leaks row2→10.0; causal correctly → NaN). Re-run `build_training_dataset.py` on the real data
   to regenerate `training_dataset.csv` and confirm counts.*
2. ✅ 🔴 **`classic_bap.py` big-M** — FIXED to `arrivals.max() + n_vessels·tau_mean·exp(3·sigma)
   + horizon` (a provably-safe upper bound on the worst single-berth pile-up). *Verified: a
   high-contention solve now uses big_m≈386 (>> the ~99 pile-up) and produces a valid
   non-overlapping schedule.*
3. ✅ 🟡 **`node.py` `tree_depth` / `realmlp.py` `depth` ignored** — FIXED (forward `depth=` to
   `DenseODSTBlock`; build `hidden_sizes` whenever `hidden_dim` OR `depth` is given). *Byte-compiles;
   needs `pytorch_tabular`/`pytabkit` installed to runtime-verify (libs absent here).*

**Also FIXED (2026-05-31):**
4. ✅ 🟡 **`build_clean_dataset` `Sitio`** is normalised numerically (int 9 / float 9.0 /
   "9" all → "Sitio 9") with NaN kept as NaN — *verified on a unit example*; **`build_report`**
   boxplot uses `set_xticklabels` instead of the matplotlib-removed `labels=` kwarg.
5. ✅ 🟡 **`test_encoders.py`** — the mislabeled "no-leak" test renamed/redocumented as a
   layout-consistency check (it never actually tested leakage); a true value-level leakage
   assertion is left as a documented TODO.

*The items in #4–5 byte-compile cleanly; they can't be run here (matplotlib / the missing
`ports_dfl.data` subpackage are absent), except the `Sitio` fix which is unit-verified.*

Per-step detail follows.

---

## `optimizers/src/bap_optim/` (step A)

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
- ✅ 🟡 **`discrete_bap.py` · `solve()` accepts `maxTimeLimit`** — FIXED. `MIPGap` is now
  `0.0` (require provable optimality) and `solve()` accepts only `optimal`. `maxTimeLimit`
  (and any other non-optimal status) now raises `RuntimeError` instead of silently returning a
  suboptimal incumbent, so the regret ≥ 0 guarantee holds exactly. The 60 s `TimeLimit` is now
  only a guard against a runaway solve — raise it if larger instances need more time.
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

---

## `prediction_models/tests/` (step D)

No assertions were weakened or changed — comments only. The **17 BAP tests still pass** after
annotation (`test_discrete_bap.py` 5 + `test_weekly_instance.py` 7 + `test_bap_windows.py` 5).

### Bugs fixed
- none (the user's rule: don't alter test logic/assertions; nothing was a clear collection-breaking bug).

### Issues reported (test quality, not changed)
- 🟡 **`test_encoders.py` · `test_train_only_target_encoding_no_leak`** — the docstring says it
  verifies no leakage, but the only assertion checks `out_train.shape[1] == out_val.shape[1]`
  (column counts). A *leaky* encoder would still pass. Either re-document it as a layout smoke
  test or add a real leakage check (e.g. val encodings equal train-derived category means).
- 🟡 **`test_discrete_bap.py`** — no `pytest.importorskip("pyepo")`/Gurobi guard, so without the
  solver stack the whole module **errors at collection** instead of skipping (the new
  `test_bap_windows.py` shows the guarded pattern). Also a hardcoded `size=5` instead of
  `small_instance.n_vessels`, and an absolute `-5.0` MIP-gap slack that won't scale to larger
  instances (prefer `0.005 * cost_fi + eps`).
- ⚪ **`test_log_target.py`** — unused `mae` import; `test_log_target_improves_or_matches_mape`
  only checks both MAPEs are finite (can't fail) — but that's intentional per its docstring.
- ⚪ **`test_tabm.py` / `test_node.py`** — `test_uses_cuda_when_available` passes the train set as
  the validation set (fine for a device smoke test); `save_load` round-trips don't assert
  `path.exists()` after `save()`.
- ⚪ **`conftest.py`** — `tiny_arrays`/`first_fold_arrays` return numpy *views* of session-scoped
  arrays; safe today (tests only read), but a future test that mutates them would corrupt shared
  state — add `.copy()` if that happens.

### Commenting
All 15 test files annotated for beginners: `test_*` auto-discovery, `@pytest.fixture` +
fixtures-as-arguments + scope, bare `assert`, `pytest.approx`, `np.testing.assert_allclose`,
`pytest.raises`, `pytest.importorskip`/`mark.skipif`/`mark.slow`, `tmp_path`, and the
arrange–act–assert structure.

---

## repo-root `src/` — data-build scripts (step F)

### Bugs fixed
- none — nothing was a clear, safe in-place fix; all items below are judgment calls / depend on
  the (absent) source data, so they are reported.

### Issues reported (not changed)
- 🔴 **`build_training_dataset.py` · subtle leakage in historical features** — the expanding
  per-vessel windows are ordered by `first_mooring_datetime` (when a visit *starts*), but the
  target `estadia_sitio_hours` is only known at `last_unmooring_datetime` (when it *ends*).
  `shift(1)` correctly drops the current row, but a *prior* visit that was still at berth when the
  current visit began can contribute an outcome that wasn't yet knowable → mild look-ahead leakage
  in `vessel_avg/median/std/last_berth_stay` and the group features. **This is the most important
  finding of the review** for a prediction/DFL project. Suggested: only include prior visits whose
  `last_unmooring_datetime` precedes the current `first_mooring_datetime` (or anchor the windows on
  unmooring). Worth validating empirically.
- 🟡 **`build_training_dataset.py`** — `clean_data` filters `estadia_sitio_hours < 2`/`> 500` but
  rows with a **NaN** target compare False on both and survive into training; add
  `dropna(subset=["estadia_sitio_hours"])` if NaN targets should be excluded.
  `tiempo_en_puerto_hours` is never cleaned.
- 🟡 **`build_clean_dataset.py` · `normalise_terminal_and_site`** — `df["Sitio"].astype(str)`
  only handles an integer `9`; if pandas loads `Sitio` as float (any NaN in the column), `9.0`
  stringifies to `"9.0"`, so the `== "9"` normalization silently never fires. Also `.astype(str)`
  turns missing `Sitio` into the literal text `"nan"` in the CSV. Normalize numerically
  (`pd.to_numeric(...) == 9`) and guard NaN.
- 🟡 **`build_clean_dataset.py` · `fix_ordering_violations` V2** — the anomaly anchors
  (`Última espía atraque`, `Fecha práctico desatraque`) aren't null-checked; a `NaT` anchor makes
  `NaN >= NaN` False and silently routes to the dispatch branch, imputing from `NaT`.
- ⚪ **`download_external_data.py`** — `pd.concat(all_dfs)` would raise on an empty list (can't
  happen with the current hardcoded `range(2020, 2026)`, but would if dates change); the year loop
  ignores `START_DATE` (always starts at 2020); unused `sys`/`json` imports.
- ⚪ **`generate_pdfs.py`** — `DATA_DIR` is never created (`doc.build()` → `FileNotFoundError` if
  `data/` is missing); hardcoded counts ("5,597 rows | 44 columns") can drift; dead `W`/`LIGHT_BLUE`/`TA_LEFT`.
- ⚪ **`port_regions.py`** — docstring says "132 origin / 110 destination ports" but the dict has
  170 unique keys (overlapping sets); region label "Argentina" also covers Uruguay ports
  (intentional per the section header — confirm downstream).

### Commenting
All 5 build scripts annotated: pandas data-cleaning/feature idioms (`read_excel`, boolean masks,
`.dt`/`Timedelta`, `.groupby().transform`, `expanding().shift(1)` anti-leakage, `np.select`),
the `requests` HTTP-fetch loop, reportlab table assembly, and the dict-mapping fallback.
