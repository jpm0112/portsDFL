"""Synthetic, classical-style DBAP instance generator.

The existing :mod:`bap_optim.discrete_bap` MILP follows Cordeau et
al. (2005). This module provides a synthetic instance generator that
mimics standard benchmark structure (Poisson-style arrivals, three-class
priority weights, lognormal service times) and exposes the difficulty
knobs that decide whether DFL can outperform PtO:

  - ``contention`` (= total work / total berth capacity): higher means
    more vessels compete for fewer berth-hours, so ordering decisions
    matter more and the DFL loss has a sharper signal.
  - ``weight_dist``: three-class {1, 2, 5} (Cordeau-style) gives a clear
    "high-priority" subset. The DFL loss weighs cost by weight, so the
    predictor can learn to spend its accuracy budget on the heavy ships.
  - ``noise_std``: how noisy the features-to-τ mapping is. With zero
    noise, PtO can already predict τ perfectly and DFL has nothing to
    add. With more noise, neither method is perfect and DFL's choice of
    *which* errors to make becomes valuable.
  - Including the priority weight as one of the features lets DFL
    (but not PtO, whose MSE is weight-agnostic) bias predictions to
    favour high-priority ships.

All instances in one batch share the same :class:`BAPInstance` descriptor
(arrivals, weights, ``big_m``) so the existing DFL trainer in
``train/dfl_blackbox.py`` can reuse the same MILP across batches.
"""

# `from __future__ import annotations` makes Python treat every type hint below
# as plain text instead of evaluating it at runtime. That lets us write modern
# hints such as `float | None` (see `horizon` below) even on older Pythons.
from __future__ import annotations

# `dataclass` is a decorator (a `@name` line written above a class) that
# auto-generates boilerplate — __init__, __repr__, == — for a class whose only
# job is to hold data. See the `@dataclass` use on `ClassicProblem` below.
from dataclasses import dataclass

import numpy as np

# Re-exported from discrete_bap (which itself re-exports it from instance.py).
# `BAPInstance` is the frozen/immutable descriptor the MILP solver consumes.
from bap_optim.discrete_bap import BAPInstance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ClassicProblem:
    """A bundle of synthetic instances that share one ``BAPInstance``.

    Fields:
        instance:   shared :class:`BAPInstance` (arrivals/weights/big_m).
        X_train:    (n_train, n_vessels, n_features) training features.
        tau_train:  (n_train, n_vessels) ground-truth service times.
        X_val:      (n_val, n_vessels, n_features) validation features.
        tau_val:    (n_val, n_vessels) ground-truth service times.
        feature_names: list of names for each feature column.
        meta:       diagnostics (contention, horizon, expected_total_work, …).
    """

    # Each line below is a "field": a name with a type hint. The @dataclass
    # decorator turns these into constructor arguments automatically, in this
    # order, e.g. ClassicProblem(instance=..., X_train=..., ...). No __init__
    # is written by hand. `np.ndarray` = a numpy array; `list[str]` = a list of
    # strings; `dict` = a Python dictionary (key -> value).
    instance: BAPInstance
    X_train: np.ndarray
    tau_train: np.ndarray
    X_val: np.ndarray
    tau_val: np.ndarray
    feature_names: list[str]
    meta: dict


# The lone `*` as the first parameter makes EVERY argument after it
# "keyword-only": callers MUST name them, e.g. make_classic_problem(seed=3),
# never make_classic_problem(8, 3, ...). This guards against silently passing
# values to the wrong knob when there are many similar numeric arguments.
# Each `name: type = default` gives a type hint plus a default value, so all of
# these are optional. `-> ClassicProblem` is the return-type hint.
def make_classic_problem(
    *,
    n_vessels: int = 8,
    n_berths: int = 3,
    n_train: int = 80,
    n_val: int = 40,
    contention: float = 1.0,
    weight_dist: str = "three_class",
    arrival: str = "uniform",
    tau_mean: float = 10.0,
    tau_sigma: float = 0.5,
    noise_std: float = 0.4,
    n_noise_features: int = 8,
    include_weight_feature: bool = True,
    horizon: float | None = None,
    seed: int = 0,
) -> ClassicProblem:
    """Generate a synthetic DBAP problem (one batch of instances).

    Args:
        n_vessels:    ships per instance.
        n_berths:     berths in the port.
        n_train:      number of training instances.
        n_val:        number of validation instances.
        contention:   :math:`E[\\tau]\\cdot N / (M\\cdot H)`. Setting horizon
            from this when ``horizon`` is ``None``. ``1.0`` means all
            berth-hours are used in expectation. ``> 1`` means oversubscribed.
        weight_dist:  ``"three_class"`` (1, 2, 5 at probabilities
            0.6/0.3/0.1, Cordeau-style), ``"uniform"`` (all 1), or
            ``"lognormal"`` (matches the original ``generate_bap_instance``).
        arrival:      ``"uniform"`` over :math:`[0, 0.7H]`, or
            ``"poisson"`` (arrivals from a Poisson process truncated to
            :math:`[0, 0.7H]`).
        tau_mean:     mean of the lognormal :math:`\\tau` distribution.
        tau_sigma:    sigma of the underlying normal in log-space; sets τ
            spread (≈ 50 % CV at 0.5).
        noise_std:    standard deviation of the noise added to the
            informative feature, in units of ``tau_mean``.
        n_noise_features:  number of pure-noise features (so the
            predictor must learn to ignore them).
        include_weight_feature: if True, the per-vessel priority weight
            is included as a feature — this is what lets DFL outperform
            PtO when weights and τ are uncorrelated.
        horizon:      if given, fixes the horizon and ignores ``contention``.
        seed:         seed for the RNG; controls every random choice.

    Returns:
        A :class:`ClassicProblem` with train / val instances.

    Notes:
        - All instances share the same arrivals and weights, but each has
          its own τ vector. This matches what
          :func:`ports_dfl.train.dfl_blackbox.train_dfl_blackbox` expects.
        - Features have the structure:
              column 0 = noisy estimate of τ (informative)
              column 1 = priority weight of the vessel (if enabled)
              column 2..K = irrelevant Gaussian noise
          Feature names are returned for traceability.
    """
    # One seeded random generator drives EVERY random draw below, so the whole
    # batch is reproducible from `seed` alone. `default_rng` is numpy's modern
    # RNG (preferred over the legacy np.random.* global functions).
    rng = np.random.default_rng(seed)
    # Generate train + val together as one block, then slice them apart at the
    # end. This guarantees train and val are drawn from the identical process.
    n_instances = n_train + n_val

    # ---- horizon from contention ------------------------------------------
    # `horizon is None` is the sentinel "caller didn't fix the horizon": derive
    # it from the contention knob instead. (Passing a horizon overrides this.)
    if horizon is None:
        # E[total work per instance] = tau_mean * n_vessels
        # Capacity per instance      = n_berths * horizon
        # contention = E[work] / capacity  ->  horizon = E[work] / (M * contention)
        # Bigger contention => smaller horizon => more crowding => sharper DFL signal.
        horizon = float(tau_mean * n_vessels / (n_berths * contention))

    # ---- service times ----------------------------------------------------
    # Lognormal with E[tau] = tau_mean.  mu = ln(mean) - sigma^2 / 2.
    # (A lognormal's mean is exp(mu + sigma^2/2); we invert that to hit tau_mean
    # exactly regardless of the chosen sigma/spread.)
    mu = np.log(tau_mean) - tau_sigma ** 2 / 2.0
    # Draw the TRUE service times: one (n_instances, n_vessels) matrix. Each
    # instance gets its own τ vector, but all share the arrivals/weights below.
    # `.astype(np.float32)` downcasts to 32-bit floats to match the torch model.
    tau_all = rng.lognormal(
        mean=mu, sigma=tau_sigma, size=(n_instances, n_vessels)
    ).astype(np.float32)

    # ---- weights (shared across instances) -------------------------------
    # Priority weights are drawn ONCE and reused for every instance (the MILP
    # objective weights completion time by these). Three named strategies:
    if weight_dist == "three_class":
        # Cordeau-style {1, 2, 5} priorities with probabilities 0.6/0.3/0.1:
        # mostly low-priority ships, a rare heavy "priority" subset.
        weights = rng.choice(
            [1.0, 2.0, 5.0], size=n_vessels, p=[0.6, 0.3, 0.1]
        ).astype(np.float32)
    elif weight_dist == "uniform":
        # All vessels equally important — removes the prioritization signal.
        weights = np.ones(n_vessels, dtype=np.float32)
    elif weight_dist == "lognormal":
        w = rng.lognormal(mean=0.0, sigma=0.7, size=n_vessels).astype(np.float32)
        # Divide by the mean so weights average ~1 (a stable objective scale).
        weights = (w / w.mean()).astype(np.float32)
    else:
        # `!r` inside an f-string inserts the repr() of the value (quotes shown),
        # making the bad input easy to spot in the error message.
        raise ValueError(f"Unknown weight_dist: {weight_dist!r}")

    # ---- arrivals (shared) -----------------------------------------------
    # Arrivals are also drawn once and reused across instances. They are kept
    # inside the first 70% of the horizon so the port stays busy (ships pile up
    # rather than trickle in evenly across the whole window).
    if arrival == "uniform":
        # `np.sort` returns arrivals in increasing time order (a tidy timeline).
        arrivals = np.sort(
            rng.uniform(0.0, 0.7 * horizon, size=n_vessels)
        ).astype(np.float32)
    elif arrival == "poisson":
        # Inter-arrival times exponential with rate set so E[max arrival]
        # is around 0.7H. The truncation to [0, 0.7H] is enforced afterwards.
        rate = float(n_vessels) / (0.7 * horizon)
        # Exponential gaps between consecutive arrivals = a Poisson process.
        inter = rng.exponential(scale=1.0 / rate, size=n_vessels)
        # `np.cumsum` turns the gaps into running arrival timestamps; because
        # every gap is positive the result is already sorted ascending.
        cum = np.cumsum(inter)
        # Rescale so the largest arrival lands close to 0.7 H. `max(cum[-1], 1e-6)`
        # avoids a divide-by-zero if every gap happened to be ~0.
        cum = cum * (0.7 * horizon / max(cum[-1], 1e-6))
        # `np.clip` caps values into [0, 0.7H] elementwise (a final safety net).
        arrivals = np.clip(cum, 0.0, 0.7 * horizon).astype(np.float32)
    else:
        raise ValueError(f"Unknown arrival pattern: {arrival!r}")

    # big-M for the MILP's precedence constraints: a number large enough that
    # the constraint s[j] >= s[i]+tau[i] - M*(1-z) is slack whenever z=0.
    big_m = float(horizon + 4.0 * tau_mean + arrivals.max())
    # Build the shared, immutable instance descriptor the solver consumes.
    instance = BAPInstance(
        n_vessels=n_vessels,
        n_berths=n_berths,
        arrivals=arrivals,
        weights=weights,
        big_m=big_m,
    )

    # ---- features ---------------------------------------------------------
    # Informative feature: noisy estimate of tau. With noise_std=0.4, the
    # signal-to-noise ratio leaves a real loss floor for both PtO and DFL.
    # `standard_normal` draws from N(0, 1); we scale it by (noise_std * tau_mean)
    # so noise_std is expressed in units of the mean service time.
    noise = rng.standard_normal(size=(n_instances, n_vessels)).astype(np.float32)
    # The model sees the truth blurred by noise; it must learn to denoise it.
    informative = tau_all + (noise_std * tau_mean) * noise

    # We assemble feature "columns" in a list, then concatenate along the last
    # axis at the end. `informative[..., None]` adds a trailing length-1 axis:
    # shape (n_inst, n_vessels) -> (n_inst, n_vessels, 1) so it stacks cleanly.
    feature_columns = [informative[..., None]]
    feature_names = ["tau_noisy"]

    if include_weight_feature:
        # Expose each vessel's priority weight as a feature. `weights[None, :, None]`
        # reshapes (n_vessels,) -> (1, n_vessels, 1); `np.broadcast_to` then
        # virtually repeats it across all instances WITHOUT copying memory, and
        # `.astype` makes a real writable copy (broadcast views are read-only).
        # This column is what lets DFL favour heavy ships even when PtO (MSE,
        # weight-blind) cannot.
        weight_broadcast = np.broadcast_to(
            weights[None, :, None], (n_instances, n_vessels, 1)
        ).astype(np.float32)
        feature_columns.append(weight_broadcast)
        feature_names.append("weight")

    if n_noise_features > 0:
        # Pure-noise decoy columns the predictor must learn to ignore.
        noise_features = rng.standard_normal(
            size=(n_instances, n_vessels, n_noise_features)
        ).astype(np.float32)
        feature_columns.append(noise_features)
        # This is a LIST COMPREHENSION: build ["noise_0", "noise_1", ...] in one
        # line. `extend` appends all of them to feature_names (vs append = one).
        feature_names.extend([f"noise_{k}" for k in range(n_noise_features)])

    # Glue the columns side by side along the last axis (axis=-1) into the final
    # feature tensor of shape (n_instances, n_vessels, total_features).
    X_all = np.concatenate(feature_columns, axis=-1).astype(np.float32)

    # ---- train / val split -----------------------------------------------
    # Slice the combined block back into train/val. `[:n_train]` = first n_train
    # rows; `[n_train:]` = everything after. Because instances were generated
    # i.i.d. and never shuffled by label, this split has no data leakage: the
    # same noise draw never appears in both halves.
    X_train = X_all[:n_train]
    X_val = X_all[n_train:]
    tau_train = tau_all[:n_train]
    tau_val = tau_all[n_train:]

    # Diagnostics bundle: pure bookkeeping for logging/sanity checks, not used
    # by the solver. `int(...)`/`float(...)` coerce numpy scalars to plain
    # Python numbers so the dict serialises cleanly (e.g. to JSON).
    meta = {
        "horizon": horizon,
        "contention": float(contention),
        "expected_total_work_per_instance": float(tau_mean * n_vessels),
        "berth_capacity_per_instance": float(n_berths * horizon),
        "tau_mean": float(tau_mean),
        "tau_sigma": float(tau_sigma),
        "noise_std": float(noise_std),
        "n_features": int(X_all.shape[-1]),
        "n_vessels": int(n_vessels),
        "n_berths": int(n_berths),
        "weight_dist": weight_dist,
        "arrival": arrival,
        "seed": int(seed),
    }

    return ClassicProblem(
        instance=instance,
        X_train=X_train,
        tau_train=tau_train,
        X_val=X_val,
        tau_val=tau_val,
        feature_names=feature_names,
        meta=meta,
    )
