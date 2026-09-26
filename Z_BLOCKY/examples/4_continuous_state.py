"""Delayed gyro sampling over continuously integrated satellite dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Z_BLOCKY import AsynchronousScheduler
from Z_BLOCKY.examples.satellite_case import build_case


def run_one(history: str, duration_s: float, seed: int):
    simulation, controller = build_case()
    result = AsynchronousScheduler().run(simulation, duration_s, seed=seed, history=history)
    return result, controller


def benchmark(duration_s: float, seed: int, repetitions: int) -> None:
    print(f"\nBenchmark: {duration_s:g} simulated seconds, {repetitions} runs per history")
    records = {}
    for history in ("dense", "cache"):
        times = []
        peaks = []
        for _ in range(repetitions):
            tracemalloc.start()
            started = time.perf_counter()
            result, _ = run_one(history, duration_s, seed)
            times.append(time.perf_counter() - started)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        records[history] = result
        print(
            f"  {history:5s}: median {statistics.median(times) * 1e3:7.2f} ms, "
            f"peak traced Python memory {max(peaks) / 1024:7.1f} KiB, "
            f"max retained history items {result.max_history_items['satellite']:3d}, "
            f"gyro evaluations {result.execution_counts['satellite.gyro']}"
        )
    assert np.allclose(records["dense"].final_states["satellite"], records["cache"].final_states["satellite"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    for history in ("dense", "cache"):
        result, controller = run_one(history, 3.0, args.seed)
        print(f"\n{history} history; controller (release, capture, angular rate):")
        for capture in result.captures:
            if 0.89 <= capture.sampled_at_s <= 0.96:
                status = "available" if capture.completed_at_s <= 1.0 else "still pending"
                print(f"  capture at {capture.sampled_at_s:.2f} s completes at "
                      f"{capture.completed_at_s:.3f} s: {status} at 1.00 s")
        for release, capture, rate in controller.observations:
            print(f"  {release:4.2f} s reads {capture:4.2f} s sample, rate {rate:.6f}")
        print(f"  gyro model calls: {result.execution_counts['satellite.gyro']} / 61 nominal captures")
        print(f"  final state: {result.final_states['satellite']}")

    simulation, _ = build_case()
    diagram = simulation.diagram(output=str(Path(__file__).with_name("4_continuous_state")))
    print(f"\nBlock diagram: {diagram}")
    result = AsynchronousScheduler().run(simulation, 3.0, seed=args.seed, history="cache")
    png, svg = simulation.timeline(
        result,
        output=str(Path(__file__).with_name("4_continuous_state_timeline")),
        title="Satellite dynamics and delayed gyro sampling",
    )
    print(f"Timing diagram: {png} and {svg}")
    benchmark(args.duration, args.seed, args.repetitions)


if __name__ == "__main__":
    main()
