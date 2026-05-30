"""Deterministic week-long berth-allocation planner.

Slices ONE planning week (pre-solve), builds a single ``BAPInstance``, solves
the discrete BAP with hard no-wait windows for priority "service" vessels, and
reports an inspectable schedule: a per-vessel table, operational KPIs, a
berth×time Gantt chart, and a schedule CSV.

This is the deterministic counterpart to the DFL training pipeline: it takes
service times τ as given (true values, or a model's predictions plugged in via
the builder's ``tau`` hook) and produces the actual weekly berth plan.

Examples
--------
  # No proprietary data needed — fabricated but realistic week:
  python scripts/plan_week.py --synthetic --n-vessels 18 --n-services 2 --slack-hours 0

  # Real data (once data/training_dataset.csv is available):
  python scripts/plan_week.py --source ../data/training_dataset.csv \
      --week-start 2025-03-03 --services 9301234,9305678 --slack-hours 0

  # Soft windows (keeps an over-booked week feasible, penalising lateness):
  python scripts/plan_week.py --synthetic --n-services 5 --soft
"""

# `from __future__ import annotations` makes all type hints (the ": int", "-> None"
# parts) be treated as plain text, not evaluated. This lets newer hint syntax like
# `int | None` work even on older Python versions. It must be the first real line.
from __future__ import annotations

import argparse  # standard library: parses command-line options like --synthetic
import sys
from pathlib import Path  # object-oriented file paths (cleaner than string paths)

import numpy as np

# Make the project's own code importable. `__file__` is THIS file's path; `.resolve()`
# turns it absolute; `.parents[1]` goes up two folders (scripts/ -> prediction_models/);
# then we append "src". Inserting at index 0 means Python searches here FIRST for imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# These imports only work because of the sys.path line above (the package lives in src/).
from ports_dfl.optim.discrete_bap import DiscreteBAP, extract_decision
from ports_dfl.optim.schedule import assemble_schedule, compute_kpis
from ports_dfl.optim.weekly_instance import (
    build_weekly_instance,
    generate_synthetic_weekly_instance,
)


# A function whose name starts with "_" is a convention meaning "internal/private".
# `-> argparse.Namespace` is a type hint: this returns a Namespace object holding the
# parsed options (accessed later as args.synthetic, args.week_start, etc.).
def _parse_args() -> argparse.Namespace:
    # ArgumentParser is the CLI builder. __doc__ is this file's top docstring (reused as
    # help text); RawDescriptionHelpFormatter keeps the docstring's line breaks intact.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # A mutually-exclusive group: the user must give EXACTLY ONE of --source / --synthetic
    # (required=True forces one; the group forbids using both at once).
    src = p.add_mutually_exclusive_group(required=True)
    # add_argument defines one flag. type=str converts the value to a string. The string
    # is the --help description. The flag "--source" becomes the attribute args.source.
    src.add_argument("--source", type=str, help="Path to a calls CSV (training_dataset.csv schema).")
    # action="store_true": a switch with no value. Present -> args.synthetic is True;
    # absent -> False. (Note the dash in "--n-vessels" becomes underscore: args.n_vessels.)
    src.add_argument("--synthetic", action="store_true", help="Fabricate a realistic week (no data needed).")

    # default=None means: if the user omits this flag, the attribute is set to None.
    p.add_argument("--week-start", type=str, default=None, help="Planning week start, e.g. 2025-03-03 (real data).")
    p.add_argument("--week-days", type=int, default=7, help="Planning window length in days (default 7).")
    p.add_argument("--services", type=str, default="none",
                   help="Real data: 'none' or comma-separated vessel ids to treat as priority services.")
    p.add_argument("--slack-hours", type=float, default=0.0,
                   help="Service latest-start slack lᵣ = aᵣ + slack (0 ⇒ no waiting).")
    p.add_argument("--base-weight", type=float, default=1.0)
    p.add_argument("--service-weight", type=float, default=3.0)
    p.add_argument("--soft", action="store_true",
                   help="Use soft (penalised) windows instead of hard ones (keeps over-booked weeks feasible).")
    p.add_argument("--penalty-weight", type=float, default=1000.0, help="Soft-window tardiness penalty.")

    # Synthetic-only knobs.
    p.add_argument("--n-vessels", type=int, default=18)
    p.add_argument("--n-services", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)  # seed makes the random week reproducible

    p.add_argument("--solver", type=str, default="gurobi")
    p.add_argument("--out", type=str, default=None, help="Output dir (default: results/plan_week).")
    p.add_argument("--no-plot", action="store_true", help="Skip the Gantt chart (no matplotlib).")
    # parse_args() reads the real command line and returns the filled Namespace.
    return p.parse_args()


# Decides which builder to call based on the flags, returning a "bundle" object
# (the instance plus arrays describing each vessel). No return type hint here.
def _build_bundle(args):
    # If --synthetic was given, fabricate a week instead of reading a CSV.
    if args.synthetic:
        return generate_synthetic_weekly_instance(
            n_vessels=args.n_vessels,
            week_days=args.week_days,
            seed=args.seed,
            n_services=args.n_services,
            service_slack_hours=args.slack_hours,
            base_weight=args.base_weight,
            service_weight=args.service_weight,
        )
    # Real-data path requires a week start date; SystemExit cleanly aborts with a message.
    if not args.week_start:
        raise SystemExit("--week-start is required with --source.")
    sel = None
    # Only build a selector list if the user passed actual ids (not "none"/empty).
    if args.services and args.services.lower() not in ("none", ""):
        # List comprehension: split "a,b,c" on commas, strip whitespace off each piece,
        # and keep only non-empty pieces. Equivalent to a small for-loop building a list.
        sel = [s.strip() for s in args.services.split(",") if s.strip()]
    return build_weekly_instance(
        args.source,
        args.week_start,
        week_days=args.week_days,
        service_selector=sel,
        service_slack_hours=args.slack_hours,
        base_weight=args.base_weight,
        service_weight=args.service_weight,
    )


# `rows: list[dict]` hints a list of dictionaries (one per vessel). Prints a text table.
def _print_schedule(rows: list[dict]) -> None:
    # f-string ("f" prefix) lets you embed values in {}. Inside the braces, ">10" pads
    # to width 10 right-aligned, "<15" pads left-aligned; this builds a fixed-width header.
    hdr = f"{'id':>10}  {'type':<15} {'berth':<10} {'arr':>7} {'start':>7} {'wait':>6} {'τ':>6} {'finish':>7} {'svc':>3} {'win':>3}"
    print(hdr)
    print("-" * len(hdr))  # a divider line as long as the header
    for r in rows:
        # "Y" if this vessel is a priority service, else blank. (Inline if/else expression.)
        svc = "Y" if r["is_service"] else ""
        # Only services have a window: blank for ordinary vessels, "ok"/"VIOL" otherwise.
        win = "" if not r["is_service"] else ("ok" if r["window_ok"] else "VIOL")
        # In format specs, ".1f" = 1 decimal place; "<15.15" = left-align AND truncate to
        # 15 chars. Adjacent f-strings on separate lines are auto-joined into one string.
        print(
            f"{str(r['vessel_id']):>10}  {r['vessel_type']:<15.15} {r['berth']:<10.10} "
            f"{r['arrival_h']:>7.1f} {r['start_h']:>7.1f} {r['wait_h']:>6.1f} "
            f"{r['tau_h']:>6.1f} {r['finish_h']:>7.1f} {svc:>3} {win:>3}"
        )


def _print_kpis(kpis: dict) -> None:
    print("\n=== KPIs ===")
    print(f"  vessels / berths / services : {kpis['n_vessels']} / {kpis['n_berths']} / {kpis['n_services']}")
    print(f"  makespan (h)                : {kpis['makespan_h']:.1f}")
    print(f"  mean / max wait (h)         : {kpis['mean_wait_h']:.1f} / {kpis['max_wait_h']:.1f}")
    print(f"  service mean / max wait (h) : {kpis['service_mean_wait_h']:.2f} / {kpis['service_max_wait_h']:.2f}")
    print(f"  all services no-wait        : {kpis['all_services_no_wait']}")
    print(f"  window violations           : {kpis['window_violations']}")
    # ":.2%" formats a fraction as a percentage with 2 decimals (e.g. 0.5 -> "50.00%").
    print(f"  mean berth utilization      : {kpis['mean_berth_utilization']:.2%}")
    # .items() yields (key, value) pairs from a dict; we unpack them into name, u.
    for name, u in kpis["berth_utilization"].items():
        print(f"      {name:<12}: {u:.2%}")


def _report_infeasible(bundle) -> None:
    print("\n!!! INFEASIBLE under hard windows !!!")
    print("The service (priority, no-wait) vessels cannot all be berthed within their")
    print("windows given berth-type compatibility. Service vessels in this week:\n")
    # np.flatnonzero returns the integer positions where is_service is True (the services).
    svc_idx = np.flatnonzero(bundle.is_service)
    for i in svc_idx:
        # For this service, list every berth it is allowed to use. range(n) is 0..n-1.
        # int(i) converts the numpy integer to a plain Python int the API expects.
        compat_berths = [bundle.berths[b].name for b in range(bundle.n_berths)
                         if bundle.instance.compatible(int(i), b)]
        # "!s" inside the braces forces str() conversion before applying the ">10" width.
        print(f"  id={bundle.vessel_ids[i]!s:>10}  type={bundle.vessel_types[i]:<15} "
              f"arr={bundle.arrivals_h[i]:.1f}h  latest_start={bundle.latest_start_h[i]:.1f}h  "
              f"berths={compat_berths}")
    # Heuristic: flag berth-type groups where simultaneous services exceed capacity.
    print("\nHint: increase --slack-hours, drop/retime a conflicting service, or use")
    print("--soft to keep the week feasible while penalising window lateness.")


# Draws a berth x time Gantt chart and saves it as a PNG at out_path.
def _gantt(rows: list[dict], bundle, out_path: Path, title: str) -> None:
    # Import inside the function so the script still runs (minus the chart) if matplotlib
    # is missing. try/except catches the ImportError and skips plotting gracefully.
    try:
        import matplotlib
        matplotlib.use("Agg")  # "Agg" = headless backend: write to a file, no GUI window
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch  # colored boxes for the legend
    except ImportError:
        print("(matplotlib not available — skipping Gantt chart)")
        return  # bail out early; the rest of the program continues

    berth_names = [b.name for b in bundle.berths]  # one row per berth on the y-axis
    # Dict comprehension mapping each berth name -> its row index (0,1,2,...).
    # enumerate gives (index, value) pairs as it loops.
    y_of = {name: k for k, name in enumerate(berth_names)}
    # subplots() returns the figure and its axes; height grows with the number of berths.
    fig, ax = plt.subplots(figsize=(12, 1.1 + 0.6 * len(berth_names)))
    for r in rows:
        # .get(key, default) looks up the berth's row; -1 if the berth name is unknown.
        y = y_of.get(r["berth"], -1)
        if y < 0:
            continue  # skip vessels whose berth isn't on the chart
        # Red for priority services, blue for ordinary vessels (hex color codes).
        color = "#B23A48" if r["is_service"] else "#2C6FA8"
        # barh = horizontal bar: a box starting at start_h, tau_h hours wide, on row y.
        ax.barh(y, r["tau_h"], left=r["start_h"], height=0.6,
                color=color, edgecolor="black", linewidth=0.5, alpha=0.9)
        # Label each bar with the vessel id, centered on the bar.
        ax.text(r["start_h"] + r["tau_h"] / 2, y, str(r["vessel_id"]),
                ha="center", va="center", fontsize=6, color="white")
        # arrival tick: a short dotted vertical line marking when the vessel arrived.
        ax.plot([r["arrival_h"], r["arrival_h"]], [y - 0.35, y + 0.35],
                color="black", linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(berth_names)))
    ax.set_yticklabels(berth_names)
    ax.set_xlabel("hours from week start")
    ax.set_title(title)
    ax.legend(handles=[Patch(color="#B23A48", label="service (no-wait)"),
                       Patch(color="#2C6FA8", label="ordinary"),
                       Patch(facecolor="white", edgecolor="black", label="| arrival (dotted)")],
              loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()  # auto-adjust spacing so labels don't get cut off
    fig.savefig(out_path, dpi=140)
    plt.close(fig)  # free the figure's memory now that it's written to disk
    print(f"  Gantt chart -> {out_path}")


# Writes the schedule rows to a CSV file.
def _write_csv(rows: list[dict], out_path: Path) -> None:
    import csv
    # FIX: guard against an empty schedule. rows[0] below would raise IndexError if the
    # solver produced no rows; bail out with a message instead of crashing.
    if not rows:
        print("  (no rows to write — skipping schedule CSV)")
        return
    # `with open(...) as f` opens the file and guarantees it is closed afterwards, even
    # on error. newline="" is the documented requirement for the csv module on Windows
    # (prevents blank lines between rows).
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # DictWriter writes dicts as rows; fieldnames (the column headers) come from the
        # keys of the first row. list(...) turns the dict_keys view into a real list.
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()  # write the header line first
        writer.writerows(rows)  # then one line per dict
    print(f"  schedule CSV -> {out_path}")


# `-> None` means this function returns nothing useful; it orchestrates everything.
def main() -> None:
    args = _parse_args()  # read CLI options
    bundle = _build_bundle(args)  # build the weekly problem (synthetic or from CSV)
    inst = bundle.instance  # the BAPInstance the solver will work on

    print(f"Weekly DBAP instance: source={bundle.source}")
    print(f"  week=[{bundle.week_start} .. {bundle.week_end})  "
          f"N={inst.n_vessels} vessels, B={inst.n_berths} berths, "
          f"{int(bundle.is_service.sum())} services, "
          f"windows={'soft' if args.soft else 'hard'}")

    # Build the optimization model. hard_windows is the opposite of --soft: by default
    # windows are HARD (must be met); --soft turns them into penalized SOFT windows.
    model = DiscreteBAP(inst, solver_name=args.solver,
                        hard_windows=not args.soft, penalty_weight=args.penalty_weight)
    model.setObj(bundle.tau_h)  # set the objective coefficients (service times tau)
    try:
        # solve() returns the chosen start times and the objective value. If the problem
        # is infeasible the solver raises RuntimeError, which we catch below.
        starts, obj = model.solve()
    except RuntimeError as e:
        # "as e" binds the exception so we can print its message.
        print(f"\nSolver error: {e}")
        _report_infeasible(bundle)  # explain WHY it failed and suggest fixes
        raise SystemExit(1)  # exit code 1 signals failure to the calling shell/script

    # extract_decision returns (assignment, order); the leading _ names the order part as
    # "unused on purpose" (a common Python convention for values we intentionally ignore).
    assignment, _order = extract_decision(model)
    rows = assemble_schedule(bundle, starts, assignment)  # turn the solution into table rows
    # horizon_h = total planning hours, used to compute berth utilization percentages.
    kpis = compute_kpis(bundle, starts, assignment, horizon_h=args.week_days * 24.0)

    print(f"\nSolved. objective (weighted completion time{' + penalty' if args.soft else ''}) = {obj:.1f}\n")
    _print_schedule(rows)
    _print_kpis(kpis)

    # Use the user's --out if given, otherwise default to <project>/results/plan_week.
    out_dir = Path(args.out) if args.out else (Path(__file__).resolve().parents[1] / "results" / "plan_week")
    # Create the folder (and any missing parents); exist_ok=True means "no error if it
    # already exists". The "/" operator joins Path pieces (out_dir / "schedule.csv").
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "schedule.csv")
    if not args.no_plot:  # skip the chart if --no-plot was passed
        title = f"Weekly berth plan ({bundle.week_start}, {'soft' if args.soft else 'hard'} windows)"
        _gantt(rows, bundle, out_dir / "gantt.png", title)

    # Non-zero exit if any service vessel waited under hard windows (should not happen).
    if not args.soft and not kpis["all_services_no_wait"]:
        print("\nWARNING: a service vessel waited despite hard windows — investigate.")
        raise SystemExit(2)  # exit code 2 distinguishes this from the solver-error exit 1


# This block runs only when the file is executed directly (python plan_week.py), NOT when
# it is imported as a module. It's the standard Python program entry point.
if __name__ == "__main__":
    main()
