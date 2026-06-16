"""Benchmark wall-clock solve time for DiscreteBAP across instance sizes.

Grid:
  n_vessels  : 5, 8, 12, 16, 20, 25, 30  (+ 40, 50 if 30 finishes under 30s)
  n_berths   : max(2, round(n_vessels/5))
  channel    : OFF (channel_time=None) and ON (channel_time=2.0)
  objective  : "waiting" and "idle"
  seeds      : 3 per cell (0, 1, 2)
  horizon    : n_vessels * 8

Timing: constructor build time and solve() wall-clock reported separately.
A solve that raises RuntimeError (timeout or infeasibility) is recorded as
">60s" and causes that (channel, objective) combo to skip larger sizes.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np

# Make sure the editable install is findable even if PYTHONPATH isn't set.
sys.path.insert(0, "C:/Users/juanp/Desktop/repos/portsDFL/optimizers/src")

from bap_optim.discrete_bap import DiscreteBAP, generate_bap_instance

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
BASE_SIZES = [5, 8, 12, 16, 20, 25, 30]
EXTENDED_SIZES = [40, 50]
SEEDS = [0, 1, 2]
CHANNEL_SETTINGS = [False, True]   # False=OFF, True=ON (channel_time=2.0)
OBJECTIVES = ["waiting", "idle"]
ARRIVAL_DENSITY = 0.7
BUILD_TIMEOUT_S = 120          # if constructor takes longer than this, skip
SOLVE_SLOW_THRESH_S = 30.0     # if median solve > this, don't try bigger sizes
TIMEOUT_SENTINEL = ">60s"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cell(
    n_vessels: int,
    n_berths: int,
    channel_on: bool,
    objective: str,
    seeds: list[int],
) -> tuple[list[float | str], list[float]]:
    """Run `seeds` solves for one (size, channel, objective) cell.

    Returns (solve_times, build_times).
    solve_times entries are floats or the TIMEOUT_SENTINEL string.
    """
    solve_times: list[float | str] = []
    build_times: list[float] = []

    for seed in seeds:
        # --- Build instance ---
        inst = generate_bap_instance(
            n_vessels=n_vessels,
            n_berths=n_berths,
            horizon_hours=float(n_vessels * 8),
            seed=seed,
            arrival_density=ARRIVAL_DENSITY,
        )
        if channel_on:
            inst = dataclasses.replace(inst, channel_time=2.0)

        rng = np.random.default_rng(seed + 1000)
        tau = rng.uniform(8, 25, size=n_vessels).astype(np.float32)

        # --- Build model ---
        t0 = time.perf_counter()
        try:
            model = DiscreteBAP(inst, objective=objective, hard_windows=False)
        except Exception as e:
            print(f"    [BUILD ERROR] N={n_vessels} ch={channel_on} obj={objective} seed={seed}: {e}", flush=True)
            solve_times.append(TIMEOUT_SENTINEL)
            build_times.append(float("nan"))
            continue
        build_time = time.perf_counter() - t0
        build_times.append(build_time)

        model.setObj(tau)

        # --- Solve ---
        t1 = time.perf_counter()
        try:
            starts, obj_val = model.solve()
            solve_time = time.perf_counter() - t1
            solve_times.append(solve_time)
            print(
                f"    N={n_vessels:2d} B={n_berths} ch={int(channel_on)} "
                f"obj={objective:7s} seed={seed} | "
                f"build={build_time:.2f}s solve={solve_time:.3f}s val={obj_val:.1f}",
                flush=True,
            )
        except RuntimeError as e:
            elapsed = time.perf_counter() - t1
            solve_times.append(TIMEOUT_SENTINEL)
            print(
                f"    N={n_vessels:2d} B={n_berths} ch={int(channel_on)} "
                f"obj={objective:7s} seed={seed} | FAILED after {elapsed:.1f}s: {e}",
                flush=True,
            )

    return solve_times, build_times


def summarise(times: list[float | str]) -> str:
    """Return 'median (max)' string, handling timeout sentinels."""
    numeric = [t for t in times if isinstance(t, float)]
    n_timeout = sum(1 for t in times if t == TIMEOUT_SENTINEL)
    if not numeric:
        return ">60s"
    med = float(np.median(numeric))
    mx = float(np.max(numeric))
    base = f"{med:.2f} ({mx:.2f})"
    if n_timeout:
        base += f" [{n_timeout}T]"
    return base


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 72, flush=True)
    print("DiscreteBAP runtime benchmark", flush=True)
    print("=" * 72, flush=True)

    # Track whether a combo has hit a timeout/slow cell; if so, skip larger N.
    combo_skip: dict[tuple[bool, str], bool] = defaultdict(bool)

    # results[n_vessels][(channel_on, objective)] = summary_string
    results: dict[int, dict[tuple[bool, str], str]] = {}

    # Determine which sizes to run; extend only if 30 is fast.
    sizes_to_run = BASE_SIZES[:]
    add_extended = True  # tentative; we'll cut it if 30 is slow

    for n_vessels in sizes_to_run + EXTENDED_SIZES:
        if n_vessels in EXTENDED_SIZES and not add_extended:
            print(f"\n  Skipping N={n_vessels} (N=30 was slow)", flush=True)
            continue

        n_berths = max(2, round(n_vessels / 5))
        print(f"\n--- N={n_vessels}, B={n_berths} ---", flush=True)
        results[n_vessels] = {}

        for channel_on in CHANNEL_SETTINGS:
            for objective in OBJECTIVES:
                combo = (channel_on, objective)
                if combo_skip[combo]:
                    results[n_vessels][combo] = f"skip (prev timeout)"
                    print(
                        f"  ch={int(channel_on)} obj={objective:7s} -> SKIPPED",
                        flush=True,
                    )
                    continue

                solve_times, build_times = run_cell(
                    n_vessels, n_berths, channel_on, objective, SEEDS
                )
                summary = summarise(solve_times)
                results[n_vessels][combo] = summary

                # Check if we should skip larger sizes for this combo.
                numeric_solves = [t for t in solve_times if isinstance(t, float)]
                has_timeout = any(t == TIMEOUT_SENTINEL for t in solve_times)
                if has_timeout:
                    combo_skip[combo] = True
                    print(f"  -> marking combo (ch={int(channel_on)}, {objective}) as SKIP", flush=True)
                elif numeric_solves and float(np.median(numeric_solves)) > SOLVE_SLOW_THRESH_S:
                    combo_skip[combo] = True
                    print(f"  -> slow (median>{SOLVE_SLOW_THRESH_S}s), marking SKIP", flush=True)

        # After N=30, decide whether to try extended sizes.
        if n_vessels == 30:
            # Check all combos for N=30 — if any timed out, skip extension.
            all_30 = results.get(30, {})
            any_timeout_at_30 = any(
                TIMEOUT_SENTINEL in v or "skip" in v
                for v in all_30.values()
            )
            slow_at_30 = any(
                isinstance(v, str) and v.split()[0] != "skip" and v.split()[0] != ">60s"
                and float(v.split()[0]) > SOLVE_SLOW_THRESH_S
                for v in all_30.values()
                if isinstance(v, str) and v[0].isdigit()
            )
            if any_timeout_at_30 or slow_at_30:
                add_extended = False
                print("\n  N=30 had timeouts or was slow — skipping N=40,50", flush=True)
            else:
                print("\n  N=30 solved quickly — will try N=40,50", flush=True)

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    combos = [
        (False, "waiting"),
        (False, "idle"),
        (True,  "waiting"),
        (True,  "idle"),
    ]
    col_headers = [
        "ch=OFF / waiting",
        "ch=OFF / idle",
        "ch=ON  / waiting",
        "ch=ON  / idle",
    ]

    print("\n\n" + "=" * 90, flush=True)
    print("RESULTS TABLE  (median solve-s, max in parens; [T]=seeds with timeout)", flush=True)
    print("=" * 90, flush=True)

    # Header
    header = f"{'N (B)':>10} | " + " | ".join(f"{h:>20}" for h in col_headers)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for n_vessels in sorted(results.keys()):
        n_berths = max(2, round(n_vessels / 5))
        row_label = f"{n_vessels} (B={n_berths})"
        cells = []
        for combo in combos:
            cells.append(results[n_vessels].get(combo, "N/A"))
        row = f"{row_label:>10} | " + " | ".join(f"{c:>20}" for c in cells)
        print(row, flush=True)

    print("=" * 90, flush=True)
    print("\nBenchmark complete.", flush=True)


if __name__ == "__main__":
    main()
