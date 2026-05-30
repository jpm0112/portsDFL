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

# `from __future__ import annotations` makes every type hint in this file be
# stored as plain text (not evaluated at import). That lets us write modern
# hints like `int | None` and reference classes before they are defined,
# even on older Python versions.
from __future__ import annotations

import numpy as np
import pyomo.environ as pyo  # Pyomo: a Python library for building math-optimization models
from pyepo import EPO  # NOTE: imported but unused in this file (see review)
from pyepo.model.omo.omomodel import optOmoModel  # base class we inherit from
from pyomo.opt import TerminationCondition  # enum describing how the solver finished

# BAPInstance lives in a dependency-light module so instance builders/tests can
# import it without the solver stack. Re-exported here for backward
# compatibility (``from bap_optim.discrete_bap import BAPInstance``).
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
    # A seeded random-number generator: same `seed` -> same instance (reproducible).
    rng = np.random.default_rng(seed)
    # Squeeze all arrivals into a fraction of the horizon so berths stay busy.
    arrival_window = horizon_hours * arrival_density
    # Draw N arrival times uniformly, sort them, and downcast to float32 to
    # match the dtypes the rest of the pipeline uses. `.astype` returns a copy.
    arrivals = np.sort(rng.uniform(0, arrival_window, size=n_vessels)).astype(np.float32)
    # Log-normal weights -> mostly near 1, with a few large "VIP" vessels.
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=n_vessels).astype(np.float32)
    # Normalise so the average weight is 1 (keeps objective magnitudes comparable
    # across instances of different sizes). This is a vectorised numpy op: it
    # divides every element by the scalar mean in one shot, no Python loop.
    weights = weights / weights.mean()
    # big-M: a number guaranteed larger than any realistic schedule span, used to
    # "switch off" precedence constraints (see the big-M constraint below).
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

# `class DiscreteBAP(optOmoModel):` declares a class that INHERITS from
# `optOmoModel` (PyEPO's base optimization-model class). Inheritance means
# DiscreteBAP automatically gets all of optOmoModel's methods/attributes and
# can add or override its own. We override `_getModel`, `setObj`, `solve`, etc.
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

    # `__init__` is the constructor: Python runs it automatically when you write
    # `DiscreteBAP(instance, ...)`. `self` is the new object being built (Python
    # passes it implicitly). `-> None` is a type hint saying the constructor
    # returns nothing. The args after `self` have type hints (e.g.
    # `solver_name: str`) and default values (`= "gurobi"`), so callers may omit
    # them.
    def __init__(
        self,
        instance: BAPInstance,
        solver_name: str = "gurobi",
        hard_windows: bool = True,
        penalty_weight: float = 1000.0,
    ) -> None:
        # Stash arguments on the object so other methods can read them via `self`.
        self.instance = instance
        # These must be set BEFORE super().__init__(), because the base
        # __init__ calls self._getModel() (which reads self.hard_windows) and
        # because PyEPO's multiprocessing path reconstructs the model from
        # same-named instance attributes (getArgs). Store every __init__ arg.
        self.solver_name = solver_name
        # `bool(...)`/`float(...)` coerce the inputs to a definite type so later
        # `if self.hard_windows` checks behave predictably.
        self.hard_windows = bool(hard_windows)
        self.penalty_weight = float(penalty_weight)
        # Service vessels with a finite window (populated in _getModel). The
        # `: list[int]` annotation documents that this holds a list of ints.
        self._service_idx: list[int] = []

        # `super().__init__(...)` calls the PARENT class's constructor
        # (optOmoModel). This is essential: the parent sets up the Pyomo model
        # (by calling our `_getModel`), the solver factory, and PyEPO bookkeeping.
        # We do it AFTER storing the attributes above because `_getModel` reads
        # them (e.g. self.hard_windows).
        super().__init__(solver=solver_name)

        # The base class ``optOmoModel.__init__`` installs a placeholder
        # ``Objective(expr=0)`` AFTER ``_getModel`` returns. Replace it with
        # the real DBAP objective (which references the mutable ``tau`` Param,
        # so τ updates propagate without rebuilding).
        self._model.del_component(self._model.obj)
        # Build the objective expression: total weighted completion time,
        # Σ wᵢ·(sᵢ + τᵢ). `sum(... for i in ...)` is a generator expression fed
        # to Python's sum(); because the terms are Pyomo objects, the result is
        # a symbolic Pyomo expression (not a number). `m.tau[i]` is the MUTABLE
        # service-time Param, so the same expression re-evaluates after τ updates.
        obj_expr = sum(
            self._model.weights[i] * (self._model.s[i] + self._model.tau[i])
            for i in self._model.I
        )
        # Soft-window mode: add the tardiness penalty term. `self._service_idx`
        # is truthy only if there is at least one windowed service vessel.
        if not self.hard_windows and self._service_idx:
            obj_expr = obj_expr + self.penalty_weight * sum(
                self._model.tard[r] for r in self._service_idx
            )
        # Attach the real objective; `sense=pyo.minimize` tells the solver to
        # minimise it (vs pyo.maximize).
        self._model.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        # Configure Gurobi-friendly options uniformly. Other solvers may
        # ignore unrecognised keys; that's acceptable. `hasattr(obj, "options")`
        # checks whether the solver factory object has an `.options` attribute
        # before we touch it (avoids an AttributeError on solvers that don't).
        if hasattr(self._solverfac, "options"):
            self._solverfac.options["MIPGap"] = 0.005     # stop within 0.5% of optimal
            self._solverfac.options["TimeLimit"] = 60     # give up after 60 seconds
            self._solverfac.options["OutputFlag"] = 0     # silence solver logging

    # `@property` turns this method into a read-only attribute: callers write
    # `model.num_cost` (no parentheses) and Python runs this function behind the
    # scenes. PyEPO reads it to learn the length of the predicted cost vector.
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
        # Short local aliases keep the model-building code readable.
        inst = self.instance
        N = inst.n_vessels
        B = inst.n_berths
        a = inst.arrivals
        w = inst.weights

        # Compatible (vessel, berth) pairs. Fail loudly if a vessel has no
        # compatible berth — that instance is structurally infeasible.
        for i in range(N):
            # `compatible_berths(i)` returns a list; an empty list is falsy.
            if not inst.compatible_berths(i):
                # f-string (the f"..." prefix): embeds {i} into the message text.
                raise ValueError(
                    f"Vessel {i} has no compatible berth (berth_compat row is all-False)."
                )
        # List comprehension building every allowed (vessel, berth) pair. Read it
        # as: "for each i, for each b, keep (i,b) only if the vessel may use it".
        # This is why x is SPARSE — incompatible pairs get no variable at all.
        xib_pairs = [(i, b) for i in range(N) for b in range(B) if inst.compatible(i, b)]

        # A ConcreteModel is the Pyomo container that holds all sets, params,
        # variables, constraints and the objective. We hang everything off `m`.
        m = pyo.ConcreteModel("dbap")

        # --- Sets ---------------------------------------------------------
        # RangeSet(0, N-1) is the integer set {0, 1, ..., N-1} (vessel indices).
        m.I = pyo.RangeSet(0, N - 1)
        m.B = pyo.RangeSet(0, B - 1)
        # Sparse assignment index: only compatible (i, b). `dimen=2` declares
        # that each member of this Set is a 2-tuple.
        m.XIB = pyo.Set(initialize=xib_pairs, dimen=2)
        # Precedence triples (i,j,b), i≠j, only where BOTH i and j may use b
        # (a pair that can never share a berth needs no sequencing).
        ijb = [
            (i, j, b)
            for i in range(N)
            for j in range(N)
            if i != j  # no self-precedence
            for b in range(B)
            if inst.compatible(i, b) and inst.compatible(j, b)
        ]
        m.IJB = pyo.Set(initialize=ijb, dimen=3)

        # --- Parameters (data) -------------------------------------------
        # A Pyomo Param is fixed input data (not a decision). Indexed over m.I,
        # initialised from a dict comprehension {vessel_index: value}.
        m.weights = pyo.Param(
            m.I, initialize={i: float(w[i]) for i in range(N)}
        )
        m.arrivals = pyo.Param(
            m.I, initialize={i: float(a[i]) for i in range(N)}
        )
        # τ is mutable — DFL/PtO updates this on every setObj. `mutable=True`
        # lets us change the values after the model is built WITHOUT rebuilding
        # the constraints/objective that reference it (just call set_value).
        m.tau = pyo.Param(m.I, mutable=True, initialize=0.0)
        # A scalar (un-indexed) Param holding the big-M constant.
        m.big_m = pyo.Param(initialize=float(inst.big_m))

        # Service vessels carrying a finite latest-start window. `is ... not None`
        # filters out vessels whose window is missing/infinite.
        self._service_idx = [
            i for i in range(N) if inst.is_service(i) and inst.latest(i) is not None
        ]

        # --- Decision variables ------------------------------------------
        # `pyo.Var` declares variables the solver chooses. domain=pyo.Binary
        # forces each x[i,b] to 0 or 1: "is vessel i assigned to berth b?".
        # Indexed over the sparse XIB set, so only compatible pairs get a var.
        m.x = pyo.Var(m.XIB, domain=pyo.Binary)

        # Start time bounds: lower = arrival; upper = latest-start window for
        # service vessels in HARD mode, else unbounded above. Pyomo calls this
        # rule once per index i, passing the model (mm) and the index; it must
        # return a (low, high) tuple. `hi = None` means "no upper bound".
        def _s_bounds(mm, i):
            lo = float(a[i])
            hi = None
            if self.hard_windows and inst.is_service(i):
                li = inst.latest(i)
                if li is not None:
                    hi = float(li)  # hard window: cannot start after lᵢ
            return (lo, hi)

        # s[i] is a continuous start time, ≥ 0, with the bounds rule above.
        m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=_s_bounds)
        # z[i,j,b] = 1 means "i is served before j at berth b" (sequencing).
        m.z = pyo.Var(m.IJB, domain=pyo.Binary)

        # Soft-window mode: instead of a hard upper bound, allow lateness but
        # penalise it. tard[r] ≥ 0 captures how far past the window r starts.
        if not self.hard_windows and self._service_idx:
            m.S = pyo.Set(initialize=self._service_idx, dimen=1)
            m.tard = pyo.Var(m.S, domain=pyo.NonNegativeReals)
            # Constraint linking tard to the overshoot. `rule=lambda mm, r: ...`
            # is an anonymous one-line function Pyomo calls per index r; it
            # returns the inequality tard[r] ≥ s[r] − lᵣ. Combined with tard ≥ 0
            # and the +penalty in the objective, the solver sets
            # tard[r] = max(0, s[r] − lᵣ).
            m.tard_con = pyo.Constraint(
                m.S,
                rule=lambda mm, r: mm.tard[r] >= mm.s[r] - float(inst.latest(r)),
            )

        # --- Constraints --------------------------------------------------
        # (1) each vessel assigned to exactly one compatible berth. The rule
        # sums the assignment binaries over that vessel's allowed berths and
        # forces the total to 1.
        m.assign = pyo.Constraint(
            m.I,
            rule=lambda mm, i: sum(mm.x[i, b] for b in inst.compatible_berths(i)) == 1,
        )

        # (3, 4) sequencing logic; only build for i < j to avoid duplicates.
        # The set IJB contains both (i,j,b) and (j,i,b); these rules cover both
        # directions in one inequality, so we Skip when i >= j to not repeat it.
        def order_max_rule(mm, i, j, b):
            if i >= j:
                return pyo.Constraint.Skip  # sentinel: "don't create this one"
            # At most one ordering direction can hold for the pair at berth b.
            return mm.z[i, j, b] + mm.z[j, i, b] <= 1

        def order_min_rule(mm, i, j, b):
            if i >= j:
                return pyo.Constraint.Skip
            # If BOTH i and j are at berth b (x sum = 2), the right side is 1,
            # forcing exactly one ordering. If not both at b, right side ≤ 0 and
            # the constraint is vacuous.
            return mm.z[i, j, b] + mm.z[j, i, b] >= mm.x[i, b] + mm.x[j, b] - 1

        m.order_max = pyo.Constraint(m.IJB, rule=order_max_rule)
        m.order_min = pyo.Constraint(m.IJB, rule=order_min_rule)

        # (5) precedence with big-M; references the mutable tau Param. When
        # z[i,j,b]=1, this reads s[j] ≥ s[i] + τᵢ (j starts after i finishes).
        # When z[i,j,b]=0, the term −M makes the right side hugely negative, so
        # the constraint is trivially satisfied ("switched off"). This is the
        # classic big-M trick for activating a constraint only when a binary is on.
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
        # is wired through the mutable Param above. `x_list` is just a Python
        # list of the N start-time Var objects, in vessel order.
        x_list = [m.s[i] for i in range(N)]
        # Return a 2-tuple; the parent __init__ unpacks it into self._model / self.x.
        return m, x_list

    # ----- PyEPO interface --------------------------------------------------

    def setObj(self, c) -> None:
        """Update predicted τ values in-place on the mutable Param.

        No model rebuild — Pyomo will re-evaluate the constraint and
        objective expressions on the next solve.

        Args:
            c: predicted service-time vector, shape (N,). Numpy or torch tensor.
        """
        # Avoid a hard dependency on torch at import time. We `try` to import it
        # locally; if torch isn't installed, the `except ImportError: pass`
        # quietly skips the tensor-conversion branch (numpy input still works).
        try:
            import torch

            # If c is a torch tensor, detach it from the autograd graph, move it
            # off any GPU to CPU, and convert to a numpy array Pyomo can read.
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
        except ImportError:
            pass
        # `np.asarray` makes a numpy array without copying if already one.
        c = np.asarray(c, dtype=np.float32)
        # Validate the prediction has exactly N entries before using it.
        if c.shape != (self.instance.n_vessels,):
            raise ValueError(
                f"cost shape {c.shape} != ({self.instance.n_vessels},)"
            )
        # Push each predicted τ into the mutable Param in place. Because the
        # objective/precedence expressions reference this Param, the next solve
        # automatically uses the new values — no rebuild needed.
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
        # Run the solver. `tee=False` hides the solver's console output.
        results = self._solverfac.solve(self._model, tee=False)
        status = results.solver.termination_condition
        # Tuple of termination conditions we treat as "usable". NOTE: a plain
        # `maxTimeLimit` only implies a within-gap solution if the 0.5% MIPGap
        # was actually reached before the 60s limit — see REPORT in review.
        accepted = (
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.locallyOptimal,
        )
        # `not in` membership test against the tuple; anything else (e.g.
        # infeasible, unbounded) is an error worth surfacing to the caller.
        if status not in accepted:
            raise RuntimeError(
                f"DBAP solver returned termination_condition={status} "
                f"(N={self.instance.n_vessels}, B={self.instance.n_berths}). "
                f"In hard-window mode this usually means the service vessels "
                f"cannot all be berthed within their no-wait windows."
            )

        N = self.instance.n_vessels
        # `pyo.value(var)` pulls the numeric solved value out of a Pyomo Var.
        # Collect all N start times into a float32 numpy array.
        starts = np.array(
            [pyo.value(self._model.s[i]) for i in range(N)], dtype=np.float32
        )
        obj_val = float(pyo.value(self._model.obj))
        # Return a 2-tuple; the type hint `-> tuple[np.ndarray, float]` documents it.
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
    # Unpack the 2D array's shape into N (rows=vessels) and B (cols=berths).
    N, B = assignment.shape
    # Fill with -1 as a "not yet scheduled" sentinel; we check for leftovers below.
    starts = np.full(N, -1.0, dtype=np.float32)

    for b in range(B):
        # Vessels assigned to this berth (binary stored as float -> compare >0.5).
        at_b = [i for i in range(N) if assignment[i, b] > 0.5]
        if not at_b:
            continue  # empty berth, nothing to schedule
        # --- Topological order from z via Kahn's algorithm ---
        # Kahn's algorithm orders nodes of a DAG so every edge points "forward".
        # Here an edge i->j means "i must precede j". in_degree[j] = how many
        # predecessors j still has; successors[i] = vessels that must follow i.
        in_degree = {i: 0 for i in at_b}      # dict comprehension: {vessel: 0}
        successors = {i: [] for i in at_b}    # dict of empty lists per vessel
        for i in at_b:
            for j in at_b:
                if i != j and order[i, j, b] > 0.5:
                    successors[i].append(j)
                    in_degree[j] += 1
        # Start with vessels that have no predecessor (in_degree 0).
        ready = [i for i in at_b if in_degree[i] == 0]
        order_list: list[int] = []
        while ready:
            # Among currently-ready vessels, take the earliest arrival first.
            # `key=lambda v: arrivals[v]` sorts by each vessel's arrival time.
            ready.sort(key=lambda v: arrivals[v])  # tie-break by arrival
            v = ready.pop(0)              # remove & return the first element
            order_list.append(v)
            # Each successor loses one predecessor; when it hits 0 it's ready.
            for w in successors[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    ready.append(w)
        # If we couldn't sequence everyone, the precedence graph had a cycle.
        if len(order_list) != len(at_b):
            raise RuntimeError(f"Cycle in precedence at berth {b}: {order}")

        # Walk the order, packing vessels back-to-back. Each start is the later
        # of (its arrival) and (when the previous vessel finished). This is the
        # cascade that makes underestimated τ hurt downstream vessels.
        prev_completion = 0.0
        for v in order_list:
            s = max(float(arrivals[v]), prev_completion)
            starts[v] = s
            prev_completion = s + float(true_tau[v])

    # `.any()` is True if any element is still the -1 sentinel (unscheduled).
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
    # Re-derive feasible starts under the true service times.
    starts = derive_starts_under_true_tau(assignment, order, true_tau, arrivals)
    # `starts + true_tau` is element-wise (completion time per vessel);
    # `np.dot(weights, ...)` is the weighted sum Σ wᵢ·(sᵢ + τᵢ) in one vectorised op.
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
    # Reach into the solved Pyomo model. The leading underscore on `_model`
    # signals it's "internal" (a convention, not enforced by Python).
    m = optmodel._model

    # Build the (N, B) assignment matrix, all zeros to start.
    assignment = np.zeros((N, B), dtype=np.float32)
    for i in range(N):
        for b in range(B):
            # x is sparse over compatible (i,b); missing pairs stay 0.
            if (i, b) in m.XIB:
                # round() snaps tiny solver fractions (e.g. 0.9999) to a clean 0/1.
                assignment[i, b] = float(round(float(pyo.value(m.x[i, b]))))

    # Read the solved start times once; used below to recover the ordering.
    starts = np.array(
        [float(pyo.value(m.s[i])) for i in range(N)], dtype=np.float32
    )

    # Rebuild z from start times instead of trusting the raw (possibly slightly
    # fractional) z vars: sorting by start time gives a clean total order per
    # berth, so every earlier vessel precedes every later one.
    order = np.zeros((N, N, B), dtype=np.float32)
    for b in range(B):
        at_b = [i for i in range(N) if assignment[i, b] > 0.5]
        # Sort key is a tuple (start_time, vessel_index): ties in start time are
        # broken by index, giving one deterministic ordering.
        at_b.sort(key=lambda i: (starts[i], i))
        # `enumerate` yields (position, vessel) pairs. For each vessel, mark it
        # as preceding every vessel after it in the sorted list (slice at_b[idx+1:]).
        for idx, i in enumerate(at_b):
            for j in at_b[idx + 1:]:
                order[i, j, b] = 1.0
    return assignment, order
