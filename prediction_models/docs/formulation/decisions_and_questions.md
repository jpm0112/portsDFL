# Decisions & Open Questions — Week-long BAP MILP

A running register of every modeling choice that is **not obvious from the data**, and every
open question we still need answered. Each entry is tagged with its **PtO/DFL implication**,
because the MILP is the optimization layer that an ML predictor feeds (`τ̂`) and that
decision-focused learning differentiates through — so a choice that looks innocuous for a
one-off deterministic plan can change what the model must predict, the training target, or
whether the optimizer stays feasible.

Status legend: **OPEN** (needs the user / port), **ASSUMED** (default chosen, revisit if
wrong), **DECIDED** (settled for now).

---

## Q1. Do we know the service time windows at the moment of planning? — **OPEN**

The "service" vessels (the port's term for committed/priority calls) carry a window during
which they must be berthed without waiting. The question is whether that window is **known
input** at planning time or **itself uncertain**.

- If **known** → the window is a hard constraint parameter `lᵣ` we read in; nothing is
  predicted about it.
- If **unknown / negotiated late** → the window (or the set of which vessels are services)
  would have to be *predicted*, which adds a second prediction target and changes the DFL
  loss and the feasibility story.

**PtO/DFL implication:** decides whether the windows are fixed constraints or part of the
prediction problem. The current implementation assumes **known** windows supplied to the
builder; flag if that is wrong.

## Q2. Are vessel arrival times `aᵢ` known (firm ETAs) at planning time, or also uncertain? — **ASSUMED known**

We treat `aᵢ` as a known input and predict only the service time `τ`.

**PtO/DFL implication:** if ETAs are noisy in reality, arrival uncertainty would interact with
the no-wait service windows (a service vessel that arrives late can cascade), and we might
eventually predict arrivals too. For now arrivals are exogenous/known; revisit if port ETAs
prove unreliable.

## Q3. Which vessels are "services"? — **ASSUMED given to the planner**

We assume the set of service (priority/committed) vessels is supplied — contractual knowledge
the port holds — rather than inferred from the data.

**PtO/DFL implication:** the service set determines which vessels get the hard `sᵣ ≤ lᵣ`
constraint and elevated weight `wᵣ`, which directly shapes the schedule and therefore the
decision-quality (regret) signal DFL trains on. If services had to be inferred, that becomes
another classifier in the pipeline.

## Q4. What is the predicted/optimized quantity `τ`? — **DECIDED: `estadia_sitio_hours`**

`τᵢ` is the berth-occupation time `estadia_sitio_hours` (first mooring → last unmooring),
the project's primary service-time target.

**PtO/DFL implication:** this is the cost vector PyEPO passes to `setObj`. It enters both the
objective (`wᵢ(sᵢ+τᵢ)`) and the precedence constraints (`sⱼ ≥ sᵢ + τᵢ − M(1−z)`), so a
mis-predicted `τ` mis-orders and mis-times the whole berth — exactly the error DFL targets.

## Q5. How is vessel–berth compatibility defined? — **ASSUMED data-driven + manual override**

Compatibility is derived from the historical `(Terminal × vessel_type_group)` co-occurrence in
`clean_dataset.csv` (a type is compatible with a berth/terminal if it was historically served
there above a small threshold), with a hand-checkable override table for edge cases. This is
**not** an authoritative port berth specification.

**PtO/DFL implication:** compatibility restricts which `x[i,b]` exist, shrinking the decision
space. If the empirical matrix is wrong (a rare-but-valid assignment dropped, or a one-off
historical assignment treated as allowed), the optimizer's feasible set is mis-specified and
DFL learns against the wrong decision geometry. **Action: validate the matrix with the port.**

## Q6. How are priority weights `wᵢ` set? — **OPEN (default proposed)**

Weights are not in the data. Proposed default: service vessels get a large weight; other
vessels weighted by type or size (e.g. TRG). The exact rule must be documented and agreed.

**PtO/DFL implication:** `wᵢ` defines the objective and therefore the entire regret signal.
DFL will happily exploit whatever weighting we choose, so an arbitrary weight scheme produces
an arbitrary "decision-focused" model. This needs a principled, port-endorsed choice.

## Q7. Hard windows in the planner vs. soft windows in DFL training — **DECIDED (dual mode)**

The deterministic weekly planner uses **hard** service windows (`hard_windows=True`):
`sᵣ ≤ lᵣ`. DFL training should use the **soft** mode (`hard_windows=False`): a tardiness
penalty `Σ pᵣ·max(0, sᵣ − lᵣ)` added to the objective, or sufficiently loose slack.

**PtO/DFL implication:** the blackbox DFL trainer re-solves the MILP under arbitrary predicted
`τ̂`. A hard window can make that solve **infeasible** for a bad prediction, which crashes
training and breaks the regret definition. Soft windows keep every re-solve feasible while
still teaching the model to respect windows. This is the central PtO/DFL-driven design choice.

## Q8. Arrival basis, big-M, and one-week scope — **DECIDED**

- **Arrival basis:** `aᵢ = F. arribo` (anchorage arrival), so wait `= sᵢ − aᵢ` is the
  anchorage-to-berth wait service vessels must avoid. (`Fecha práctico atraque` is an
  alternative; this is configurable.)
- **big-M:** `M = horizon (168 h) + max plausible total service at a single berth`. Must be
  large enough not to cut feasible schedules, small enough to keep the LP relaxation tight.
- **One-week scope:** the MILP solves exactly one 7-day instance. Slicing is a pre-solve step;
  there is **no** rolling-horizon or cross-week carry-over. Vessels straddling the week
  boundary (arriving before the window but still in service at its start, or arriving near the
  end) need a documented convention — current default: include a vessel iff its `F. arribo`
  falls inside the window.

**PtO/DFL implication:** big-M and the horizon affect numerical conditioning of every solve in
the DFL loop; the one-week scope fixes the instance size `N` the predictor and optimizer see.

---

## How to use this file

Add an entry whenever you make a choice that a reader could not reconstruct from the data or
the code alone. Keep the open questions at the top of mind in meetings with the port — Q1, Q3,
Q5, and Q6 in particular gate whether the model's outputs are operationally trustworthy.
