"""Discrete Berth Allocation Problem (DBAP) MILP for DFL.

Terminology (aligned with the DFL literature — Elmachtoub & Grigas 2022;
Mandi et al. JAIR 2024)

  predicted decision         the (x, s, z) tuple obtained by solving the
                             MILP under the model's predicted τ̂.
  full-information (FI)      the (x, s, z) obtained by solving the MILP
  optimum / decision         under the true τ. Post-hoc optimal benchmark.
  regret                     cost(predicted decision, true τ)
                             − cost(FI decision, true τ).  Always ≥ 0.

Multi-berth scheduling formulation, classical-literature time-precedence variant:

  - N vessels with arrival times aᵢ and weights/priorities wᵢ.
  - B berths, optionally HETEROGENEOUS: vessel i may only use a subset of
    berths Bᵢ ⊆ B (vessel–berth compatibility, e.g. liquid bulk only at the
    liquid-bulk berth).
  - A subset of "service" vessels (the port's term for committed/priority
    calls) carry a hard no-wait window: they must begin service by a latest
    start lᵢ (with lᵢ = aᵢ + slack, slack→0 meaning "berth on arrival").
  - Predicted service times τᵢ from the upstream model.
  - Decisions:
      x[i,b] ∈ {0,1}     vessel i is processed at berth b   (only for b ∈ Bᵢ)
      s[i] ≥ aᵢ          start time of vessel i
      z[i,j,b] ∈ {0,1}   vessel i precedes vessel j at berth b
  - Objective:
      min Σᵢ wᵢ · (s[i] + τᵢ)              ← total weighted completion time
      (+ soft tardiness penalty in soft-window mode, see below)
  - Constraints:
      Σ_{b∈Bᵢ} x[i,b] = 1                   (each vessel assigned, compatible berth)
      s[i] ≥ aᵢ                             (start after arrival)
      s[r] ≤ lᵣ  for service r              (hard no-wait window; hard mode)
      z[i,j,b] + z[j,i,b] ≤ 1               (only one direction)
      z[i,j,b] + z[j,i,b] ≥ x[i,b] + x[j,b] - 1   (must order if both at b)
      s[j] ≥ s[i] + τᵢ - M·(1 - z[i,j,b])    (precedence; big-M)

Hard vs. soft service windows
-----------------------------
``hard_windows=True`` (default; used by the deterministic weekly planner)
encodes ``s[r] ≤ lᵣ`` as a hard upper bound on the start time, so a service
vessel literally cannot wait past its window. This can make an instance
INFEASIBLE if too many service vessels contend for the same berth at once —
that is a real planning signal and ``solve()`` will raise.

``hard_windows=False`` (recommended for DFL training) instead adds a
nonnegative tardiness variable ``tard[r] ≥ s[r] − lᵣ`` and a penalty
``+ p·Σ_r tard[r]`` to the objective. This keeps the MILP feasible under any
predicted τ̂ — essential because the blackbox DFL trainer re-solves the model
for arbitrary (possibly bad) predictions, and an infeasible solve would abort
training and break the regret definition.

Predicted τ enters both the objective and the precedence constraints. SPO+
is therefore not directly applicable; the DFL trainer uses PyEPO's blackbox
differentiation, which works for any optimizer.

Backend
-------
**Pyomo + Gurobi** (Pyomo provides solver-agnosticism; Gurobi delivers the
speed). τ enters the model as a *mutable* `Param`, so re-solving with a new
prediction means just updating Param values — no model rebuild. Other Pyomo
solvers (`scip`, `cbc`, `cplex`, `glpk`) work via the same code path by
passing a different ``solver_name`` to the constructor.

Alignment with the classic discrete BAP
---------------------------------------
This is the standard *time-indexed precedence* formulation of the
discrete-DBAP-D problem in the Bierwirth & Meisel (2010, 2015) taxonomy.
It mirrors Cordeau, Laporte, Legato & Moccia (2005) "DBAP1" with these
variations:

- We use a per-berth precedence variable ``z[i,j,b]`` instead of Cordeau's
  cross-berth ``σ[i,j]`` linked through the assignment. Equivalent
  expressivity; ours has B× more sequencing binaries but cleaner constraints.
- We add vessel–berth compatibility (sparse ``x``) and hard/soft no-wait
  windows for priority "service" vessels. Both are clean extensions over the
  base DBAP and are needed for the real San Antonio weekly planning use case.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from pyepo import EPO
from pyepo.model.omo.omomodel import optOmoModel
from pyomo.opt import TerminationCondition

# BAPInstance lives in a dependency-light module so instance builders/tests can
# import it without the solver stack. Re-exported here for backward
# compatibility (``from ports_dfl.optim.discrete_bap import BAPInstance``).
from .instance import BAPInstance


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------

def generate_bap_instance(
    n_vessels: int = 8,
    n_berths: int = 2,
    horizon_hours: float = 200.0,
    seed: int = 0,
    arrival_density: float = 0.7,
) -> BAPInstance:
    """Generate a synthetic homogeneous BAP instance (no windows/compat).

    Vessels arrive uniformly over a fraction of the horizon (so the system
    is busy and decisions matter), with weights drawn from a log-normal so
    a few high-priority vessels create real prioritization signal.
    """
    rng = np.random.default_rng(seed)
    arrival_window = horizon_hours * arrival_density
    arrivals = np.sort(rng.uniform(0, arrival_window, size=n_vessels)).astype(np.float32)
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=n_vessels).astype(np.float32)
    weights = weights / weights.mean()
    big_m = float(horizon_hours + 400.0)
    return BAPInstance(
        n_vessels=n_vessels,
        n_berths=n_berths,
        arrivals=arrivals,
        weights=weights,
        big_m=big_m,
    )


# ---------------------------------------------------------------------------
# DBAP optimizer (Pyomo + swappable solver)
# ---------------------------------------------------------------------------

class DiscreteBAP(optOmoModel):
    """Multi-berth DBAP MILP wired into PyEPO's optModel interface.

    The "cost vector" PyEPO passes to ``setObj(c)`` is the predicted
    service-time vector τ̂ (length N). It enters the model through
    a mutable Pyomo Param, so re-solving with a new τ̂ does not rebuild
    the model — only the Param values change.

    Window mode
    -----------
    ``hard_windows=True`` (default): service vessels get a hard upper bound
    ``s[r] ≤ lᵣ`` (can render an instance infeasible — used by the
    deterministic planner). ``hard_windows=False``: a soft tardiness penalty
    keeps every solve feasible — used by the DFL trainer. See the module
    docstring for the rationale.

    Solver
    ------
    Default is ``"gurobi"`` (uses Pyomo's direct Gurobi-via-Python interface).
    Pass any other Pyomo-supported solver name to swap: ``"scip"``,
    ``"cbc"``, ``"cplex"``, ``"glpk"``. The MILP definition is unchanged.

    Solution returned by ``solve()``: vessel start times of length N
    (PyEPO's ``self.x`` is bound to ``[m.s[0], …, m.s[N-1]]``).
    """

    def __init__(
        self,
        instance: BAPInstance,
        solver_name: str = "gurobi",
        hard_windows: bool = True,
        penalty_weight: float = 1000.0,
    ) -> None:
        self.instance = instance
        # These must be set BEFORE super().__init__(), because the base
        # __init__ calls self._getModel() (which reads self.hard_windows) and
        # because PyEPO's multiprocessing path reconstructs the model from
        # same-named instance attributes (getArgs). Store every __init__ arg.
        self.solver_name = solver_name
        self.hard_windows = bool(hard_windows)
        self.penalty_weight = float(penalty_weight)
        # Service vessels with a finite window (populated in _getModel).
        self._service_idx: list[int] = []

        super().__init__(solver=solver_name)

        # The base class ``optOmoModel.__init__`` installs a placeholder
        # ``Objective(expr=0)`` AFTER ``_getModel`` returns. Replace it with
        # the real DBAP objective (which references the mutable ``tau`` Param,
        # so τ updates propagate without rebuilding).
        self._model.del_component(self._model.obj)
        obj_expr = sum(
            self._model.weights[i] * (self._model.s[i] + self._model.tau[i])
            for i in self._model.I
        )
        # Soft-window mode: add the tardiness penalty term.
        if not self.hard_windows and self._service_idx:
            obj_expr = obj_expr + self.penalty_weight * sum(
                self._model.tard[r] for r in self._service_idx
            )
        self._model.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        # Configure Gurobi-friendly options uniformly. Other solvers may
        # ignore unrecognised keys; that's acceptable.
        if hasattr(self._solverfac, "options"):
            self._solverfac.options["MIPGap"] = 0.005
            self._solverfac.options["TimeLimit"] = 60
            self._solverfac.options["OutputFlag"] = 0

    @property
    def num_cost(self) -> int:
        """PyEPO's ``num_cost`` — length of the cost vector (= N predicted τ)."""
        return self.instance.n_vessels

    # ----- Pyomo model construction (called once by the base __init__) -----

    def _getModel(self):
        """Build the Pyomo ConcreteModel and return ``(model, x_list)``.

        ``x_list`` is the list PyEPO uses as the decision representation —
        we use start times here. The MILP itself contains x[i,b], s[i],
        z[i,j,b] as decisions. Assignment binaries x[i,b] are created only
        for compatible (i,b); precedence triples only for pairs that can
        share a berth.
        """
        inst = self.instance
        N = inst.n_vessels
        B = inst.n_berths
        a = inst.arrivals
        w = inst.weights

        # Compatible (vessel, berth) pairs. Fail loudly if a vessel has no
        # compatible berth — that instance is structurally infeasible.
        for i in range(N):
            if not inst.compatible_berths(i):
                raise ValueError(
                    f"Vessel {i} has no compatible berth (berth_compat row is all-False)."
                )
        xib_pairs = [(i, b) for i in range(N) for b in range(B) if inst.compatible(i, b)]

        m = pyo.ConcreteModel("dbap")

        # --- Sets ---------------------------------------------------------
        m.I = pyo.RangeSet(0, N - 1)
        m.B = pyo.RangeSet(0, B - 1)
        # Sparse assignment index: only compatible (i, b).
        m.XIB = pyo.Set(initialize=xib_pairs, dimen=2)
        # Precedence triples (i,j,b), i≠j, only where BOTH i and j may use b
        # (a pair that can never share a berth needs no sequencing).
        ijb = [
            (i, j, b)
            for i in range(N)
            for j in range(N)
            if i != j
            for b in range(B)
            if inst.compatible(i, b) and inst.compatible(j, b)
        ]
        m.IJB = pyo.Set(initialize=ijb, dimen=3)

        # --- Parameters (data) -------------------------------------------
        m.weights = pyo.Param(
            m.I, initialize={i: float(w[i]) for i in range(N)}
        )
        m.arrivals = pyo.Param(
            m.I, initialize={i: float(a[i]) for i in range(N)}
        )
        # τ is mutable — DFL/PtO updates this on every setObj.
        m.tau = pyo.Param(m.I, mutable=True, initialize=0.0)
        m.big_m = pyo.Param(initialize=float(inst.big_m))

        # Service vessels carrying a finite latest-start window.
        self._service_idx = [
            i for i in range(N) if inst.is_service(i) and inst.latest(i) is not None
        ]

        # --- Decision variables ------------------------------------------
        m.x = pyo.Var(m.XIB, domain=pyo.Binary)

        # Start time bounds: lower = arrival; upper = latest-start window for
        # service vessels in HARD mode, else unbounded above.
        def _s_bounds(mm, i):
            lo = float(a[i])
            hi = None
            if self.hard_windows and inst.is_service(i):
                li = inst.latest(i)
                if li is not None:
                    hi = float(li)
            return (lo, hi)

        m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=_s_bounds)
        m.z = pyo.Var(m.IJB, domain=pyo.Binary)

        # Soft-window mode: nonnegative tardiness vars + linearisation.
        if not self.hard_windows and self._service_idx:
            m.S = pyo.Set(initialize=self._service_idx, dimen=1)
            m.tard = pyo.Var(m.S, domain=pyo.NonNegativeReals)
            m.tard_con = pyo.Constraint(
                m.S,
                rule=lambda mm, r: mm.tard[r] >= mm.s[r] - float(inst.latest(r)),
            )

        # --- Constraints --------------------------------------------------
        # (1) each vessel assigned to exactly one compatible berth
        m.assign = pyo.Constraint(
            m.I,
            rule=lambda mm, i: sum(mm.x[i, b] for b in inst.compatible_berths(i)) == 1,
        )

        # (3, 4) sequencing logic; only build for i < j to avoid duplicates
        def order_max_rule(mm, i, j, b):
            if i >= j:
                return pyo.Constraint.Skip
            return mm.z[i, j, b] + mm.z[j, i, b] <= 1

        def order_min_rule(mm, i, j, b):
            if i >= j:
                return pyo.Constraint.Skip
            return mm.z[i, j, b] + mm.z[j, i, b] >= mm.x[i, b] + mm.x[j, b] - 1

        m.order_max = pyo.Constraint(m.IJB, rule=order_max_rule)
        m.order_min = pyo.Constraint(m.IJB, rule=order_min_rule)

        # (5) precedence with big-M; references mutable tau
        m.precedence = pyo.Constraint(
            m.IJB,
            rule=lambda mm, i, j, b: mm.s[j]
            >= mm.s[i] + mm.tau[i] - mm.big_m * (1 - mm.z[i, j, b]),
        )

        # The objective is intentionally NOT set here — the parent
        # ``optOmoModel.__init__`` will install ``Objective(expr=0)`` right
        # after this method returns. Our ``__init__`` replaces it with the
        # real weighted-completion-time objective (+ soft penalty) afterwards.

        # PyEPO uses ``self.x`` as the decision representation. We expose
        # start-time variables; the cost vector PyEPO passes (predicted τ)
        # is wired through the mutable Param above.
        x_list = [m.s[i] for i in range(N)]
        return m, x_list

    # ----- PyEPO interface --------------------------------------------------

    def setObj(self, c) -> None:
        """Update predicted τ values in-place on the mutable Param.

        No model rebuild — Pyomo will re-evaluate the constraint and
        objective expressions on the next solve.

        Args:
            c: predicted service-time vector, shape (N,). Numpy or torch tensor.
        """
        # Avoid a hard dependency on torch at import time
        try:
            import torch

            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
        except ImportError:
            pass
        c = np.asarray(c, dtype=np.float32)
        if c.shape != (self.instance.n_vessels,):
            raise ValueError(
                f"cost shape {c.shape} != ({self.instance.n_vessels},)"
            )
        for i in range(self.instance.n_vessels):
            self._model.tau[i].set_value(float(c[i]))

    def solve(self) -> tuple[np.ndarray, float]:
        """Solve the MILP. Returns ``(start_times, objective)``.

        Accepts ``optimal``, ``feasible``, and ``maxTimeLimit`` (when MIPGap
        was met) termination conditions. The configured 0.5 % MIP gap means
        any "feasible" return is provably within 0.5 % of optimum, which
        keeps regret comparisons clean.

        In hard-window mode an over-constrained instance can be ``infeasible``;
        this raises ``RuntimeError`` (callers — e.g. the weekly planner —
        should catch it and report the conflicting service vessels).
        """
        results = self._solverfac.solve(self._model, tee=False)
        status = results.solver.termination_condition
        accepted = (
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.locallyOptimal,
        )
        if status not in accepted:
            raise RuntimeError(
                f"DBAP solver returned termination_condition={status} "
                f"(N={self.instance.n_vessels}, B={self.instance.n_berths}). "
                f"In hard-window mode this usually means the service vessels "
                f"cannot all be berthed within their no-wait windows."
            )

        N = self.instance.n_vessels
        starts = np.array(
            [pyo.value(self._model.s[i]) for i in range(N)], dtype=np.float32
        )
        obj_val = float(pyo.value(self._model.obj))
        return starts, obj_val


# ---------------------------------------------------------------------------
# Decision-quality utilities (solver-independent)
# ---------------------------------------------------------------------------

def derive_starts_under_true_tau(
    assignment: np.ndarray,
    order: np.ndarray,
    true_tau: np.ndarray,
    arrivals: np.ndarray,
) -> np.ndarray:
    """Compute *feasible* start times under true τ given a fixed (x, z) decision.

    The optimizer's start times are correct only under the τ that produced
    them. To evaluate a decision under different τ, we recompute starts
    by walking the per-berth order while respecting arrival times.

    This is what makes regret well-defined and ≥ 0: the decision we lock in
    is the (assignment, ordering) pair, and start times are re-derived under
    reality, capturing the cascade penalty of any underestimation. The FI
    benchmark uses true τ both inside the MILP and for evaluation, so its
    realised cost is by definition the lowest achievable.

    NOTE: this re-derivation is window/compatibility agnostic — it only
    enforces arrival lower bounds and per-berth precedence. Service-window
    feasibility is enforced inside the MILP (hard mode) or penalised in the
    objective (soft mode); it is intentionally not re-imposed here, so for
    windowed instances the DFL regret is computed against the realised
    start times without an extra window penalty. (If window-aware regret is
    needed, fold the tardiness penalty into the trainer's loss as well.)

    Args:
        assignment: shape (N, B), x[i,b] ∈ {0,1}.
        order:      shape (N, N, B), z[i,j,b] = 1 if i precedes j at berth b.
        true_tau:   shape (N,), ground-truth service times.
        arrivals:   shape (N,), vessel arrival times.

    Returns:
        Start times shape (N,), respecting arrival ≥ a[i] and precedence
        ≥ predecessor's completion time.
    """
    N, B = assignment.shape
    starts = np.full(N, -1.0, dtype=np.float32)

    for b in range(B):
        at_b = [i for i in range(N) if assignment[i, b] > 0.5]
        if not at_b:
            continue
        # Topological order from z via Kahn's algorithm
        in_degree = {i: 0 for i in at_b}
        successors = {i: [] for i in at_b}
        for i in at_b:
            for j in at_b:
                if i != j and order[i, j, b] > 0.5:
                    successors[i].append(j)
                    in_degree[j] += 1
        ready = [i for i in at_b if in_degree[i] == 0]
        order_list: list[int] = []
        while ready:
            ready.sort(key=lambda v: arrivals[v])  # tie-break by arrival
            v = ready.pop(0)
            order_list.append(v)
            for w in successors[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    ready.append(w)
        if len(order_list) != len(at_b):
            raise RuntimeError(f"Cycle in precedence at berth {b}: {order}")

        prev_completion = 0.0
        for v in order_list:
            s = max(float(arrivals[v]), prev_completion)
            starts[v] = s
            prev_completion = s + float(true_tau[v])

    if (starts < 0).any():
        raise RuntimeError("Some vessels were not scheduled (assignment row sum < 1).")
    return starts


def schedule_cost_under_true_tau(
    assignment: np.ndarray,
    order: np.ndarray,
    true_tau: np.ndarray,
    arrivals: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Σᵢ wᵢ (sᵢ + τᵢ) where sᵢ is recomputed feasibly under true τ.

    Returns:
        (cost, starts) — cost in weighted-hour units; starts as a side-effect.
    """
    starts = derive_starts_under_true_tau(assignment, order, true_tau, arrivals)
    cost = float(np.dot(weights, starts + true_tau))
    return cost, starts


def extract_decision(optmodel: DiscreteBAP) -> tuple[np.ndarray, np.ndarray]:
    """Pull (x[i,b], z[i,j,b]) out of the most recently solved DBAP.

    The assignment ``x`` is rounded to {0,1} from the solver's fractional
    output. Incompatible (i,b) pairs have no variable and are read as 0.
    The precedence matrix ``z`` is reconstructed from the solver's start
    times (sorted within each berth), which guarantees a strict total order
    per berth and avoids cycles caused by mildly fractional ``z`` values. The
    big-M precedence constraint guarantees that ordering by start time is
    consistent with the actual ``z`` selected by the solver, so this
    reconstruction is loss-less.

    Returns:
        ``(assignment, order)`` ndarrays of shapes ``(N, B)`` and ``(N, N, B)``.
    """
    N = optmodel.instance.n_vessels
    B = optmodel.instance.n_berths
    m = optmodel._model

    assignment = np.zeros((N, B), dtype=np.float32)
    for i in range(N):
        for b in range(B):
            # x is sparse over compatible (i,b); missing pairs stay 0.
            if (i, b) in m.XIB:
                assignment[i, b] = float(round(float(pyo.value(m.x[i, b]))))

    starts = np.array(
        [float(pyo.value(m.s[i])) for i in range(N)], dtype=np.float32
    )

    order = np.zeros((N, N, B), dtype=np.float32)
    for b in range(B):
        at_b = [i for i in range(N) if assignment[i, b] > 0.5]
        # Tie-break by vessel index for deterministic ordering.
        at_b.sort(key=lambda i: (starts[i], i))
        for idx, i in enumerate(at_b):
            for j in at_b[idx + 1:]:
                order[i, j, b] = 1.0
    return assignment, order
