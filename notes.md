# Review notes — publication readiness (branch `review-fixes`)

Full review of the portsDFL codebase (2026-07-18): methodology, correctness,
hygiene, and over-engineering. Fixes live on branch `review-fixes` (5 commits vs
`main`). All **143 tests pass under real Gurobi** (22 optimizers + 121
prediction_models) with the WLS academic license.

---

## 1. Fixed on `review-fixes` (verified)

### M1 — Objective/loss/metric mismatch (the load-bearing fix)

**Problem.** The MILP's default objective is unweighted waiting `Σᵢ(sᵢ − aᵢ)`;
τ enters only the big-M precedence constraints, never the objective. But the DFL
loss, the regret metric, the README, and the report all scored *weighted
completion* `Σᵢ wᵢ(sᵢ + τᵢ)` — an objective the optimizer never minimises.
Consequences: (a) the full-information (FI) benchmark was not the minimiser of
the scored metric, so **regret ≥ 0 was not guaranteed** (the old `-1e-1` test
tolerances were absorbing real negative regret); (b) weights never reached the
optimizer, so DFL could not "learn to prioritise heavy vessels" through it —
the mechanism `run_dfl_synthetic.py` claims to demonstrate.

**Fix (chosen direction: align to the unweighted objective).** Dropped weights
from `schedule_cost_under_true_tau` (now `Σᵢ(sᵢ + τᵢ)`), from both DFL trainers'
loss (`(starts + tau_b).sum(dim=1).mean()`) and regret evaluation; tightened the
regret/FI-optimality test tolerances from `-1e-1` to `-1e-3`; renamed
`weighted_cost_*` summary keys to `cost_*` (run scripts, `compare.py`,
`build_report.py`); corrected README + report text.

**Why regret ≥ 0 now holds (verified two ways).**
- *Formal:* the scored cost `Σᵢ(sᵢ + τᵢ)` equals the MILP objective
  `Σᵢ(sᵢ − aᵢ)` plus the constant `Σᵢ(aᵢ + τᵢ)`. For a fixed (assignment,
  order), the greedy earliest-start packing in `derive_starts_under_true_tau`
  is the componentwise-minimal feasible start vector, so the FI solve's
  re-derived cost equals the minimum of the re-derived cost over *all*
  decisions.
- *Empirical:* brute-force enumeration of every (assignment, order) decision on
  6 random 5-vessel/2-berth instances — FI re-derived cost equalled the true
  minimum with gap 0.00e+00 in all trials; prediction-driven regret ≥ 0 always.

**Publication framing note.** The DFL mechanism is now honestly the *cascade
asymmetry* (under-predicting τ causes downstream lateness; over-predicting only
idles a berth) — not weight prioritisation. The alternative design (weighted
completion `Σᵢ wᵢ(sᵢ+τᵢ)` *inside* the MILP objective) would make weights
drive scheduling and satisfy PyEPO's linear-objective assumption, but
invalidates all committed results and requires a full re-run. Revisit before
submission if the prioritisation story matters.

### C1 — big-M did not scale with instance size
`generate_bap_instance` used `big_m = horizon + 400`, which violates the
invariant documented in `instance.py` ("≥ horizon plus the largest plausible
total service time at one berth") and can silently cut valid schedules once a
berth's cumulative completion exceeds it — the solver still reports `optimal`
on the wrongly-constrained model. Now `big_m = max(arrivals) + N·τ_ub`
(`service_time_ub` parameter, default `horizon_hours`).

### M4 — empty validation set silently disabled early stopping
`_evaluate_regret` returned `np.mean([]) = nan`; no epoch ever "improved".
Both trainers now raise `ValueError` on an empty val set.

### C3 — MIPGap=0 exactness is Gurobi-only
`MIPGap`/`TimeLimit`/`OutputFlag` are Gurobi option names; other solvers
silently ignored them, leaving their default gap (~5–10%) while `solve()`
still accepted `optimal` — noise that can flip regret negative. Options now
applied only for Gurobi; other solvers get a `UserWarning`.

### C2 — regret is channel-agnostic (documented, dormant)
`derive_starts_under_true_tau` does not re-impose the navigation channel's
single-transit serialisation. Dormant: no DFL script sets `channel_time`.
Documented in the docstring; matters only if the channel is enabled during
learning.

### Test-suite fixes
- `test_bap_windows::test_backward_compatible_no_windows_no_compat` asserted
  `obj > 0`, but the instance's true optimum waiting is exactly 0 (arrivals
  spread widely enough for a conflict-free schedule). **Pre-existing bug**,
  proven by failing identically on `main`'s code under Gurobi. Now `>= -1e-3`.
- `test_discrete_bap.py` guarded with `pytest.importorskip` (was a collection
  error without the solver stack).
- New `tests/test_dfl_train.py`: seeded end-to-end smoke test of the blackbox
  DFL loop (runs, finite traces, regret ≥ 0). The trainers previously had zero
  direct coverage.
- `summarize_folds`: single-fold std was NaN (ddof=1); falls back to ddof=0.
- `regret_relative_pct`: guarded division by a zero FI-cost mean.
- Removed dead `from pyepo import EPO`.

---

## 2. Open items — **publication blockers** (decisions needed)

These were documented (in `prediction_models/docs/REVIEW_FINDINGS.md`, Round 2)
but deliberately **not changed in code** per owner's decision. For publication
they need resolving, not just documenting:

1. **`Calado diff` suspected target leakage (🔴 highest priority).** It is the
   arrival−departure draft difference — knowable only *after* service
   (`docs/project_description.md` flags draft difference as post-berthing) —
   yet it sits in `config.ALL_FEATURES`. `predictor/features.py` defaults it to
   0 at inference and calls `covid_era` cutoffs "reverse-engineered", which is
   consistent with training having used the post-hoc value. **Action: trace how
   `data/training_dataset.csv` computed it; if post-hoc, drop the feature and
   re-tune. A reviewer who spots this will question every prediction metric.**
2. **`data/training_dataset.csv` is not reproducible.** The committed
   `data_pipeline/` builders emit a different schema (`estadia_sitio_hours`, …)
   than the models consume (`service_time_hours`, `covid_era`, cyclical
   `atraque_*`, `Calado diff`). No committed script regenerates the consumed
   CSV, so the careful leakage-safe logic in `build_training_dataset.py`
   applies to a schema the models don't use. **Action: commit the actual
   feature-engineering step (even if the raw data stays private).**
3. **"Real BAP" framing.** `run_dfl_real_bap.py` builds instances by randomly
   permuting unrelated real vessels into groups sharing one synthetic
   arrival/weight vector. τ is real; the scheduling geometry is not. **Action:
   state as a limitation ("real service times on synthetic instance geometry"),
   or build instances from actual weekly call windows.**
4. **Committed results are stale.** `dfl_report.pdf`, `best_config.json`s, and
   all regret numbers predate the M1 alignment. **Action: re-run all
   experiments on `review-fixes` before quoting any number.**
5. **Stale docs.** Top-level `README.md` still claims `ports_dfl.data` is
   missing (it exists and is wired); `REVIEW_FINDINGS.md` items 1–3 of Round 1
   are fixed but the text reads as open. **Action: sweep both before sharing.**

## 3. Known limitations (acceptable if stated)

- PyEPO's `blackboxOpt`/`perturbedOpt` assume the prediction enters a linear
  objective `cᵀz`; here τ̂ enters the *constraints* (the report's
  "predicted-constraints DFL setting" wording is now accurate). DBB is used
  outside its stated assumptions — defensible, but it is an argument to make
  explicitly in the paper, not a lint to ignore.
- `berths.py` `min_compat_count=1`: one historical co-occurrence ⇒ permanent
  vessel-type/berth compatibility. Consider a higher threshold; the
  `DEFAULT_BERTHS` catalog is marked "must be validated against the port".
- No `TimeLimit` is set for non-Gurobi solvers (C3 gating) — a pathological
  instance could run long under HiGHS/CBC. Accepted: DFL/regret experiments are
  Gurobi-only.
- `run_dfl_real_bap.py` `--n_train_instances 80` random permutations of val
  rows share the single fold-0 preprocessing — fine for the demo, but a
  publication run should use the full CV protocol.

## 4. Environment / reproducibility notes

- Gurobi: WLS academic license (LICENSEID 2449798, Auburn) installed at
  `~/gurobi.lic` (2026-07-18); the expired named-user license is backed up at
  `~/gurobi.lic.expired-20260610.bak`. Works with the pinned
  `gurobipy==11.0.2` — no env changes needed.
- `highspy` (HiGHS) was added to the `portsdfl` conda env as a license-free
  fallback solver; harmless, remove with `pip uninstall highspy` if unwanted.
- Test invocation: `python -m pytest` from `optimizers/` and
  `prediction_models/` respectively, in the `portsdfl` env.

---

*An independent multi-model code review (Opus + Gemini/Codex CLIs) of the
`review-fixes` diff is appended below when it completes.*

## 5. Independent review of the `review-fixes` diff

*(pending — will be appended)*
