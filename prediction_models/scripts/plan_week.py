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

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The optimizers (bap_optim) now live in the sibling top-level package optimizers/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimizers" / "src"))

from bap_optim.discrete_bap import DiscreteBAP, extract_channel, extract_decision
from bap_optim.schedule import assemble_schedule, compute_kpis
from bap_optim.weekly_instance import (
    build_weekly_instance,
    generate_synthetic_weekly_instance,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Require exactly one data source.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=str, help="Path to a calls CSV (training_dataset.csv schema).")
    src.add_argument("--synthetic", action="store_true", help="Fabricate a realistic week (no data needed).")

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
    p.add_argument("--channel-time", type=float, default=None,
                   help="Model a single shared navigation channel: hours each vessel needs to enter "
                        "and to exit. No two transits overlap; objective becomes weighted departure. "
                        "Omit for the berth-only model.")

    # Synthetic-only knobs.
    p.add_argument("--n-vessels", type=int, default=18)
    p.add_argument("--n-services", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--solver", type=str, default="gurobi")
    p.add_argument("--out", type=str, default=None, help="Output dir (default: results/plan_week).")
    p.add_argument("--no-plot", action="store_true", help="Skip the Gantt chart (no matplotlib).")
    return p.parse_args()


def _build_bundle(args):
    if args.synthetic:
        return generate_synthetic_weekly_instance(
            n_vessels=args.n_vessels,
            week_days=args.week_days,
            seed=args.seed,
            n_services=args.n_services,
            service_slack_hours=args.slack_hours,
            base_weight=args.base_weight,
            service_weight=args.service_weight,
            channel_time=args.channel_time,
        )
    if not args.week_start:
        raise SystemExit("--week-start is required with --source.")
    sel = None
    if args.services and args.services.lower() not in ("none", ""):
        sel = [s.strip() for s in args.services.split(",") if s.strip()]
    return build_weekly_instance(
        args.source,
        args.week_start,
        week_days=args.week_days,
        service_selector=sel,
        service_slack_hours=args.slack_hours,
        base_weight=args.base_weight,
        service_weight=args.service_weight,
        channel_time=args.channel_time,
    )


def _print_schedule(rows: list[dict]) -> None:
    # The channel columns (enter/exit/depart) only exist when a channel was modelled.
    has_channel = bool(rows) and "departure_h" in rows[0]
    hdr = f"{'id':>10}  {'type':<15} {'berth':<10} {'arr':>7} {'start':>7} {'wait':>6} {'τ':>6} {'finish':>7}"
    if has_channel:
        hdr += f" {'enter':>7} {'exit':>7} {'depart':>7}"
    hdr += f" {'svc':>3} {'win':>3}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        svc = "Y" if r["is_service"] else ""
        # Only services have a window.
        win = "" if not r["is_service"] else ("ok" if r["window_ok"] else "VIOL")
        line = (
            f"{str(r['vessel_id']):>10}  {r['vessel_type']:<15.15} {r['berth']:<10.10} "
            f"{r['arrival_h']:>7.1f} {r['start_h']:>7.1f} {r['wait_h']:>6.1f} "
            f"{r['tau_h']:>6.1f} {r['finish_h']:>7.1f}"
        )
        if has_channel:
            line += f" {r['enter_h']:>7.1f} {r['exit_h']:>7.1f} {r['departure_h']:>7.1f}"
        line += f" {svc:>3} {win:>3}"
        print(line)


def _print_kpis(kpis: dict) -> None:
    print("\n=== KPIs ===")
    print(f"  vessels / berths / services : {kpis['n_vessels']} / {kpis['n_berths']} / {kpis['n_services']}")
    print(f"  makespan (h)                : {kpis['makespan_h']:.1f}")
    print(f"  mean / max wait (h)         : {kpis['mean_wait_h']:.1f} / {kpis['max_wait_h']:.1f}")
    print(f"  service mean / max wait (h) : {kpis['service_mean_wait_h']:.2f} / {kpis['service_max_wait_h']:.2f}")
    print(f"  all services no-wait        : {kpis['all_services_no_wait']}")
    print(f"  window violations           : {kpis['window_violations']}")
    print(f"  mean berth utilization      : {kpis['mean_berth_utilization']:.2%}")
    for name, u in kpis["berth_utilization"].items():
        print(f"      {name:<12}: {u:.2%}")
    ch = kpis.get("channel")
    if ch is not None:
        print(f"  channel transit (h)         : {ch['transit_time_h']:.1f}  ({ch['n_transits']} transits)")
        print(f"  channel utilization         : {ch['utilization']:.2%}")
        print(f"  port makespan (last depart) : {ch['port_makespan_h']:.1f}")
        print(f"  channel no-overlap          : {ch['no_overlap']}  (overlaps={ch['overlaps']})")


def _report_infeasible(bundle) -> None:
    print("\n!!! INFEASIBLE under hard windows !!!")
    print("The service (priority, no-wait) vessels cannot all be berthed within their")
    print("windows given berth-type compatibility. Service vessels in this week:\n")
    svc_idx = np.flatnonzero(bundle.is_service)
    for i in svc_idx:
        compat_berths = [bundle.berths[b].name for b in range(bundle.n_berths)
                         if bundle.instance.compatible(int(i), b)]
        print(f"  id={bundle.vessel_ids[i]!s:>10}  type={bundle.vessel_types[i]:<15} "
              f"arr={bundle.arrivals_h[i]:.1f}h  latest_start={bundle.latest_start_h[i]:.1f}h  "
              f"berths={compat_berths}")
    # Heuristic: flag berth-type groups where simultaneous services exceed capacity.
    print("\nHint: increase --slack-hours, drop/retime a conflicting service, or use")
    print("--soft to keep the week feasible while penalising window lateness.")


# Draws a berth x time Gantt chart and saves it as a PNG at out_path.
def _gantt(rows: list[dict], bundle, out_path: Path, title: str) -> None:
    # Import here so the script still runs (minus the chart) if matplotlib is missing.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend: write to a file, no GUI window
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("(matplotlib not available — skipping Gantt chart)")
        return

    berth_names = [b.name for b in bundle.berths]
    y_of = {name: k for k, name in enumerate(berth_names)}
    fig, ax = plt.subplots(figsize=(12, 1.1 + 0.6 * len(berth_names)))
    for r in rows:
        y = y_of.get(r["berth"], -1)
        if y < 0:
            continue
        color = "#B23A48" if r["is_service"] else "#2C6FA8"
        ax.barh(y, r["tau_h"], left=r["start_h"], height=0.6,
                color=color, edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.text(r["start_h"] + r["tau_h"] / 2, y, str(r["vessel_id"]),
                ha="center", va="center", fontsize=6, color="white")
        # arrival tick
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Gantt chart -> {out_path}")


def _write_csv(rows: list[dict], out_path: Path) -> None:
    import csv
    # FIX: guard against an empty schedule; rows[0] below would raise IndexError.
    if not rows:
        print("  (no rows to write — skipping schedule CSV)")
        return
    # newline="" is required by the csv module on Windows (prevents blank lines).
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  schedule CSV -> {out_path}")


def main() -> None:
    args = _parse_args()
    bundle = _build_bundle(args)
    inst = bundle.instance

    print(f"Weekly DBAP instance: source={bundle.source}")
    print(f"  week=[{bundle.week_start} .. {bundle.week_end})  "
          f"N={inst.n_vessels} vessels, B={inst.n_berths} berths, "
          f"{int(bundle.is_service.sum())} services, "
          f"windows={'soft' if args.soft else 'hard'}")

    # hard_windows is the opposite of --soft: by default windows are HARD (must be met).
    model = DiscreteBAP(inst, solver_name=args.solver,
                        hard_windows=not args.soft, penalty_weight=args.penalty_weight)
    model.setObj(bundle.tau_h)
    try:
        starts, obj = model.solve()
    except RuntimeError as e:
        print(f"\nSolver error: {e}")
        _report_infeasible(bundle)
        raise SystemExit(1)

    assignment, _order = extract_decision(model)
    # ein/eout are None unless a channel was modelled (extract_channel guards that).
    ein, eout = extract_channel(model)
    rows = assemble_schedule(bundle, starts, assignment, ein=ein, eout=eout)
    kpis = compute_kpis(bundle, starts, assignment,
                        horizon_h=args.week_days * 24.0, ein=ein, eout=eout)

    objective_name = ("weighted departure" if bundle.channel_time is not None
                      else "weighted completion time")
    print(f"\nSolved. objective ({objective_name}{' + penalty' if args.soft else ''}) = {obj:.1f}\n")
    _print_schedule(rows)
    _print_kpis(kpis)

    out_dir = Path(args.out) if args.out else (Path(__file__).resolve().parents[1] / "results" / "plan_week")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "schedule.csv")
    if not args.no_plot:
        title = f"Weekly berth plan ({bundle.week_start}, {'soft' if args.soft else 'hard'} windows)"
        _gantt(rows, bundle, out_dir / "gantt.png", title)

    # Non-zero exit if any service vessel waited under hard windows (should not happen).
    if not args.soft and not kpis["all_services_no_wait"]:
        print("\nWARNING: a service vessel waited despite hard windows — investigate.")
        raise SystemExit(2)  # exit code 2 distinguishes this from the solver-error exit 1

    # Non-zero exit if two channel transits overlap (should never happen — the
    # MILP forbids it; this catches a modelling/extraction regression).
    ch = kpis.get("channel")
    if ch is not None and not ch["no_overlap"]:
        print(f"\nWARNING: {ch['overlaps']} channel-transit overlap(s) detected — investigate.")
        raise SystemExit(3)  # exit code 3 distinguishes channel violations


if __name__ == "__main__":
    main()
