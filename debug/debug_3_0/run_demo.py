"""Run and plot the four block-graph prototype demonstrations.

Examples
--------
    python -m debug.debug_3_0.run_demo baseline
    python -m debug.debug_3_0.run_demo asynchronous
    python -m debug.debug_3_0.run_demo multicore
    python -m debug.debug_3_0.run_demo gpu
    python -m debug.debug_3_0.run_demo all --show
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .prototype import (
    AsyncEventEngine,
    FixedStepEngine,
    ModelConfig,
    SimulationResult,
    build_satellite,
    default_timings,
    quat_conjugate,
    quat_multiply,
    run_cpu_monte_carlo,
)


BLOCK_ORDER = ["orbit", "attitude", "disturbances", "sensors", "estimator", "controller", "actuators", "logger"]
COLORS = dict(zip(BLOCK_ORDER, plt.cm.tab10(np.linspace(0, 1, len(BLOCK_ORDER)))))


def quaternion_error_deg(q: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0, 0.0]) if reference is None else reference
    values = np.atleast_2d(q)
    out = []
    for value in values:
        error = quat_multiply(quat_conjugate(reference), value)
        out.append(math.degrees(2 * math.atan2(np.linalg.norm(error[1:]), abs(error[0]))))
    return np.asarray(out)


def _save(fig: plt.Figure, output: Path, filename: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def _event_raster(ax: plt.Axes, result: SimulationResult, until: float) -> None:
    index = {name: i for i, name in enumerate(BLOCK_ORDER)}
    for entry in result.trace:
        if entry.start > until:
            continue
        ax.scatter(entry.start, index[entry.block], marker="|", s=85, color=COLORS[entry.block])
    ax.set_yticks(range(len(BLOCK_ORDER)), BLOCK_ORDER)
    ax.set_xlim(0, until)
    ax.set_xlabel("simulation time [s]")
    ax.set_title("Executions on the common minimum-period lattice")
    ax.grid(axis="x", alpha=0.25)


def plot_block_graph(config: ModelConfig, output: Path) -> Path:
    graph = build_satellite(config)
    positions = {
        "orbit": (0.7, 2.6), "attitude": (0.7, 1.2), "disturbances": (3.0, 2.0),
        "sensors": (5.3, 3.0), "estimator": (7.6, 3.0), "controller": (9.9, 2.0),
        "actuators": (7.6, 0.8), "logger": (5.3, 0.2),
    }
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ax.set_xlim(-0.2, 11.3)
    ax.set_ylim(-0.5, 4.0)
    ax.axis("off")
    for block in graph.blocks:
        x, y = positions[block.name]
        box = FancyBboxPatch(
            (x - 0.78, y - 0.32), 1.56, 0.64,
            boxstyle="round,pad=0.06", facecolor=COLORS[block.name], edgecolor="black", alpha=0.82,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.08, block.name, ha="center", va="center", color="white", weight="bold")
        ax.text(x, y - 0.14, f"{block.timing.period:g} s", ha="center", va="center", color="white", fontsize=9)
    for source, target, signal in graph.connections:
        if target == "logger":
            continue  # Observer fan-in would obscure the physical/FSW graph.
        sx, sy = positions[source]
        tx, ty = positions[target]
        arrow = FancyArrowPatch(
            (sx, sy), (tx, ty), arrowstyle="-|>", mutation_scale=12,
            connectionstyle="arc3,rad=0.08", color=COLORS[source], lw=1.1, alpha=0.48,
            shrinkA=38, shrinkB=38,
        )
        ax.add_patch(arrow)
    ax.text(5.5, 3.75, "Validated satellite block graph (period shown in each block)", ha="center", fontsize=15)
    ax.text(5.3, -0.25, "logger observes every signal; feedback loops consume the latest delivered value", ha="center", fontsize=9)
    return _save(fig, output, "00_validated_block_graph.png")


def baseline_demo(output: Path) -> tuple[SimulationResult, Path]:
    config = ModelConfig(duration=12.0, orbit_degree=4, orbit_integrator="rk4", attitude_integrator="rk4")
    plot_block_graph(config, output)
    result = FixedStepEngine().run(build_satellite(config))
    attitude = result.array("attitude")
    estimate = result.array("estimate")
    orbit = result.array("orbit")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(result.time, quaternion_error_deg(attitude[:, 3:7]), label="truth")
    axes[0].plot(result.time, quaternion_error_deg(estimate[:, 3:7]), label="MEKF estimate", alpha=0.8)
    axes[0].set_ylabel("attitude error [deg]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    altitude = np.linalg.norm(orbit[:, :3], axis=1) - 6_378_137.0
    axes[1].plot(result.time, altitude - altitude[0])
    axes[1].set_ylabel("altitude change [m]")
    axes[1].grid(alpha=0.25)
    _event_raster(axes[2], result, until=1.0)
    fig.suptitle(
        f"Baseline block graph — selected base tick = {result.base_tick:.3f} s, "
        f"{len(result.trace)} block calls"
    )
    return result, _save(fig, output, "01_baseline_minimum_tick.png")


def _timing_diagram(ax: plt.Axes, result: SimulationResult, until: float = 1.2) -> None:
    index = {name: i for i, name in enumerate(BLOCK_ORDER)}
    for entry in result.trace:
        if entry.start > until:
            continue
        row = index[entry.block]
        compute = max(entry.finish - entry.start, 8e-4)
        ax.broken_barh([(entry.start, compute)], (row - 0.32, 0.64), facecolors=COLORS[entry.block])
        if entry.delivery > entry.finish + 1e-12:
            ax.plot([entry.finish, entry.delivery], [row, row], color=COLORS[entry.block], lw=1.1, alpha=0.8)
            ax.scatter(entry.delivery, row, marker="o", s=10, color=COLORS[entry.block])
    ax.set_yticks(range(len(BLOCK_ORDER)), BLOCK_ORDER)
    ax.set_xlim(0, until)
    ax.set_xlabel("simulation time [s]")
    ax.set_title("Asynchronous execution: bars = compute, lines/points = output delay/delivery")
    ax.grid(axis="x", alpha=0.25)


def asynchronous_demo(output: Path) -> tuple[SimulationResult, Path]:
    config = ModelConfig(duration=12.0, timings=default_timings(asynchronous=True))
    result = AsyncEventEngine().run(build_satellite(config))
    attitude = result.array("attitude")
    estimate = result.array("estimate")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(result.time, quaternion_error_deg(attitude[:, 3:7]), label="truth")
    axes[0].plot(result.time, quaternion_error_deg(estimate[:, 3:7]), label="delayed MEKF estimate")
    axes[0].set_ylabel("attitude error [deg]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for key in ("sensors_age", "estimate_age", "control_age", "actuator_age"):
        axes[1].step(result.time, result.history[key], where="post", label=key.removesuffix("_age"))
    axes[1].set_ylabel("information age [s]")
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=0.25)
    _timing_diagram(axes[2], result)
    fig.suptitle("Fully asynchronous block graph with timestamped, delayed signals")
    return result, _save(fig, output, "02_asynchronous_timing.png")


def multicore_demo(output: Path) -> tuple[list[SimulationResult], Path]:
    config = ModelConfig(duration=12.0)
    results = run_cpu_monte_carlo(config, seeds=(101, 102, 103), workers=3)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for result, seed in zip(results, (101, 102, 103)):
        attitude = result.array("attitude")
        axes[0].plot(result.time, quaternion_error_deg(attitude[:, 3:7]), label=f"satellite {seed}")
        axes[1].plot(result.time, np.rad2deg(np.linalg.norm(attitude[:, :3], axis=1)), label=f"satellite {seed}")
    axes[0].set_ylabel("attitude error [deg]")
    axes[1].set_ylabel("body rate [deg/s]")
    axes[1].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Three independent satellite Monte Carlos — 3 CPU worker processes")
    return results, _save(fig, output, "03_multicore_three_satellites.png")


def gpu_demo(output: Path, *, allow_cpu: bool = False):
    from .gpu_backend import run_gpu_monte_carlo

    config = ModelConfig(duration=12.0)
    result = run_gpu_monte_carlo(config, seeds=(101, 102, 103), require_gpu=not allow_cpu)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for run, seed in enumerate(result.seeds):
        axes[0].plot(result.time, quaternion_error_deg(result.attitude[:, run, 3:7]), label=f"satellite {seed}")
        error = np.array([
            quaternion_error_deg(result.estimated_q[k, run], result.attitude[k, run, 3:7])[0]
            for k in range(len(result.time))
        ])
        axes[1].plot(result.time, error, label=f"satellite {seed}")
    axes[0].set_ylabel("attitude error [deg]")
    axes[1].set_ylabel("estimation error [deg]")
    axes[1].set_xlabel("simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Three-satellite batched Monte Carlo — JAX device: {result.device}")
    filename = "04_gpu_three_satellites.png" if "gpu" in result.device.lower() else "04_jax_cpu_validation.png"
    return result, _save(fig, output, filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=("baseline", "asynchronous", "multicore", "gpu", "all"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--allow-jax-cpu", action="store_true",
        help="validate the JAX backend on CPU; the plot remains labeled with the actual device",
    )
    args = parser.parse_args()
    jobs = {
        "baseline": lambda: baseline_demo(args.output),
        "asynchronous": lambda: asynchronous_demo(args.output),
        "multicore": lambda: multicore_demo(args.output),
        "gpu": lambda: gpu_demo(args.output, allow_cpu=args.allow_jax_cpu),
    }
    selected = jobs if args.demo == "all" else {args.demo: jobs[args.demo]}
    failures = 0
    for name, job in selected.items():
        try:
            result, path = job()
            wall = sum(item.wall_time for item in result) if isinstance(result, list) else result.wall_time
            print(f"{name:12s} -> {path} ({wall:.3f} s reported compute time)")
        except RuntimeError as exc:
            failures += 1
            print(f"{name:12s} -> unavailable: {exc}")
    if args.show:
        plt.show()
    else:
        plt.close("all")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
