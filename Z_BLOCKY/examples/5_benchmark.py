"""Compare scheduler runtime and block call counts on the shared satellite case."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Z_BLOCKY import AcademicScheduler, AsynchronousScheduler, BaseScheduler
from Z_BLOCKY.examples.satellite_case import build_chained_case


CASES = (
    ("Base", BaseScheduler(), "cache"),
    ("Async dense", AsynchronousScheduler(), "dense"),
    ("Async cache", AsynchronousScheduler(), "cache"),
    ("Academic", AcademicScheduler(), "cache"),
)


def run_case(scheduler, history: str, duration_s: float, seed: int):
    simulation, _ = build_chained_case()
    started = time.perf_counter()
    result = scheduler.run(simulation, duration_s, seed=seed, history=history, trace=False)
    elapsed = time.perf_counter() - started
    return elapsed, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="simulated seconds per run")
    parser.add_argument("--repetitions", type=int, default=5, help="timed runs per scheduler")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    print(
        f"Satellite chain: state source 100 Hz → gyro 100 Hz → controller 1 Hz, "
        f"{args.duration:g} simulated seconds, {args.repetitions} repetitions"
    )
    rows = []
    for label, scheduler, history in CASES:
        # Warm SciPy/NumPy and the Python execution path; exclude this run.
        run_case(scheduler, history, min(args.duration, 1.0), args.seed)
        timings = []
        result = None
        for _ in range(args.repetitions):
            elapsed, result = run_case(scheduler, history, args.duration, args.seed)
            timings.append(elapsed)
        assert result is not None
        # Tracing allocations changes wall times, so measure memory in an
        # additional run outside the timed repetitions.
        gc.collect()
        tracemalloc.start()
        _, memory_result = run_case(scheduler, history, args.duration, args.seed)
        gc.collect()
        retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append((label, statistics.median(timings), result, retained_bytes, peak_bytes, memory_result))

    dense_state = rows[1][2].final_states["satellite"]
    cache_state = rows[2][2].final_states["satellite"]
    if not np.allclose(dense_state, cache_state, rtol=1e-9, atol=1e-11):
        raise RuntimeError("dense and cached asynchronous histories changed the final plant state")

    baseline = rows[0][1]
    print("\nMedian wall time and separate peak Python allocation run (lower is better):")
    print(f"{'Scheduler':<14} {'Time ms':>9} {'vs Base':>8} {'Peak KiB':>10} {'End KiB':>9} {'Truth':>8} {'Gyro':>8} {'Control':>9} {'State hist.':>12}")
    for label, elapsed, result, retained_bytes, peak_bytes, memory_result in rows:
        truth_count = result.execution_counts["satellite.truth_rate"]
        gyro_count = result.execution_counts["satellite.gyro"]
        control_count = result.execution_counts["satellite.controller"]
        history_count = (result.max_history_items or {}).get("satellite", 0)
        speedup = baseline / elapsed if elapsed else float("inf")
        print(
            f"{label:<14} {elapsed * 1e3:>9.2f} {speedup:>7.2f}x "
            f"{peak_bytes / 1024:>10.1f} {retained_bytes / 1024:>9.1f} "
            f"{truth_count:>8} {gyro_count:>8} "
            f"{control_count:>9} {history_count:>12}"
        )
        if memory_result.execution_counts != result.execution_counts:
            raise RuntimeError(f"inconsistent counts between timed and memory runs: {label}")
    print("Peak/End KiB are traced Python allocations; native SciPy/NumPy memory is not fully tracked.")


if __name__ == "__main__":
    main()
