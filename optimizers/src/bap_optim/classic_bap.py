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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Re-exported from discrete_bap (which itself re-exports it from instance.py).
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

    instance: BAPInstance
    X_train: np.ndarray
    tau_train: np.ndarray
    X_val: np.ndarray
    tau_val: np.ndarray
    feature_names: list[str]
    meta: dict


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
    rng = np.random.default_rng(seed)
    # Generate train + val together as one block, then slice them apart at the
    # end, so both are drawn from the identical process.
    n_instances = n_train + n_val

    # ---- horizon from contention ------------------------------------------
    # horizon is None => derive it from the contention knob (passing a horizon
    # overrides this).
    if horizon is None:
        # E[total work per instance] = tau_mean * n_vessels
        # Capacity per instance      = n_berths * horizon
        # contention = E[work] / capacity  ->  horizon = E[work] / (M * contention)
        # Bigger contention => smaller horizon => more crowding => sharper DFL signal.
        horizon = float(tau_mean * n_vessels / (n_berths * contention))

    # ---- service times ----------------------------------------------------
    # Lognormal with E[tau] = tau_mean. The mean is exp(mu + sigma^2/2), so we
    # invert that for mu to hit tau_mean exactly regardless of sigma.
    mu = np.log(tau_mean) - tau_sigma ** 2 / 2.0
    # Each instance gets its own τ vector, but all share the arrivals/weights below.
    tau_all = rng.lognormal(
        mean=mu, sigma=tau_sigma, size=(n_instances, n_vessels)
    ).astype(np.float32)

    # ---- weights (shared across instances) -------------------------------
    # Drawn ONCE and reused for every instance. Three named strategies:
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
        raise ValueError(f"Unknown weight_dist: {weight_dist!r}")

    # ---- arrivals (shared) -----------------------------------------------
    # Drawn once and reused. Kept inside the first 70% of the horizon so the
    # port stays busy (ships pile up rather than trickle in evenly).
    if arrival == "uniform":
        arrivals = np.sort(
            rng.uniform(0.0, 0.7 * horizon, size=n_vessels)
        ).astype(np.float32)
    elif arrival == "poisson":
        # Exponential inter-arrival gaps (= a Poisson process), rate set so
        # E[max arrival] ≈ 0.7H; truncation to [0, 0.7H] enforced afterwards.
        rate = float(n_vessels) / (0.7 * horizon)
        inter = rng.exponential(scale=1.0 / rate, size=n_vessels)
        # cumsum of positive gaps is already sorted ascending.
        cum = np.cumsum(inter)
        # Rescale so the largest arrival lands near 0.7H; max(..., 1e-6) avoids /0.
        cum = cum * (0.7 * horizon / max(cum[-1], 1e-6))
        arrivals = np.clip(cum, 0.0, 0.7 * horizon).astype(np.float32)
    else:
        raise ValueError(f"Unknown arrival pattern: {arrival!r}")

    # big-M must exceed the worst-case s[i]+tau[i]-s[j] — the longest completion
    # when all N vessels stack at one berth (arrivals.max() + sum of tau).
    # FIX: the old `horizon + 4*tau_mean + arrivals.max()` could fall below that
    # pile-up for high-contention instances, letting a too-small M spuriously
    # force precedence even when z=0 (corrupting schedules / the DFL regret).
    # Size M from a high per-vessel tau quantile (~exp(3*sigma)) times N. A larger
    # M never cuts a feasible schedule; it only loosens the LP relaxation.
    big_m = float(arrivals.max() + n_vessels * tau_mean * np.exp(3.0 * tau_sigma) + horizon)
    instance = BAPInstance(
        n_vessels=n_vessels,
        n_berths=n_berths,
        arrivals=arrivals,
        weights=weights,
        big_m=big_m,
    )

    # ---- features ---------------------------------------------------------
    # Informative feature: noisy estimate of tau. noise_std is in units of
    # tau_mean; at 0.4 the SNR leaves a real loss floor for both PtO and DFL.
    noise = rng.standard_normal(size=(n_instances, n_vessels)).astype(np.float32)
    informative = tau_all + (noise_std * tau_mean) * noise

    feature_columns = [informative[..., None]]
    feature_names = ["tau_noisy"]

    if include_weight_feature:
        # broadcast_to repeats the weights across instances WITHOUT copying;
        # .astype makes a writable copy (broadcast views are read-only). This
        # column lets DFL favour heavy ships even when PtO (MSE, weight-blind) cannot.
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
        feature_names.extend([f"noise_{k}" for k in range(n_noise_features)])

    X_all = np.concatenate(feature_columns, axis=-1).astype(np.float32)

    # ---- train / val split -----------------------------------------------
    # No data leakage: instances are i.i.d. and never shuffled by label, so the
    # same noise draw never appears in both halves.
    X_train = X_all[:n_train]
    X_val = X_all[n_train:]
    tau_train = tau_all[:n_train]
    tau_val = tau_all[n_train:]

    # Diagnostics: bookkeeping for logging/sanity checks, not used by the solver.
    # int/float coerce numpy scalars so the dict serialises cleanly (e.g. JSON).
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
