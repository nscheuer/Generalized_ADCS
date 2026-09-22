"""Run the hierarchical fixed-step architecture demonstrations.

Examples::

    python -m debug.debug_3_0.run_blocks fixed
    python -m debug.debug_3_0.run_blocks subblocks
    python -m debug.debug_3_0.run_blocks async
    python -m debug.debug_3_0.run_blocks cpu-mc
    python -m debug.debug_3_0.run_blocks gpu-mc
    python -m debug.debug_3_0.run_blocks power
    python -m debug.debug_3_0.run_blocks benchmark
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .batch_backend import BatchResult, run_jax_monte_carlo, run_numba_monte_carlo
from .block_engine import AsyncEventEngine, CompositeBlock, FixedStepEngine, RunResult, compile_fixed
from .satellite_model import Estimate, PowerState, SatelliteModel, build_model


def _save(fig, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.savefig(path, dpi=170, bbox_inches="tight")
    return path


def _q_error_deg(q):
    q = np.asarray(q)
    return np.rad2deg(2 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0])))


def plot_graph(satellite: SatelliteModel, output: Path, filename: str) -> Path:
    plan = compile_fixed(satellite.model)
    positions = {
        "orbit_propagation": (1.0, 3.4), "attitude_propagation": (1.0, 1.2),
        "disturbances": (3.1, 2.3), "sensors": (5.0, 3.6),
        "sensors.gyro": (4.35, 3.85), "sensors.magnetometer": (5.25, 3.85),
        "sensors.sun_sensor": (4.35, 3.15), "sensors.packet": (5.25, 3.15),
        "estimator": (7.0, 3.5), "controller": (9.0, 2.5), "actuators": (7.0, 1.0),
        "power.solar_panels": (3.8, 0.45), "power.battery": (5.3, 0.45),
        "recorder": (9.1, 0.45),
    }
    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    ax.set(xlim=(0, 10.3), ylim=(-0.2, 4.8))
    ax.axis("off")
    paths = {item.block: item.block.path for item in plan.blocks}
    nested = any(path.startswith("sensors.") for path in paths.values())
    powered = any(path.startswith("power.") for path in paths.values())
    if nested:
        ax.add_patch(Rectangle((3.75, 2.72), 2.15, 1.65, facecolor="#eaf2fb", edgecolor="#3973ac", lw=1.8))
        ax.text(3.82, 4.23, "sensors", color="#24547d", weight="bold")
    if powered:
        ax.add_patch(Rectangle((3.05, 0.05), 3.0, 0.9, facecolor="#fff3d8", edgecolor="#ad7d13", lw=1.8))
        ax.text(3.12, 0.82, "power", color="#7b5708", weight="bold")

    colors = plt.cm.tab20(np.linspace(0, 1, len(plan.blocks)))
    color = {item.block: colors[i] for i, item in enumerate(plan.blocks)}
    for item in plan.blocks:
        block = item.block
        x, y = positions[block.path]
        width = 1.35 if "." not in block.path else 0.78
        height = 0.62 if "." not in block.path else 0.47
        box = FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle="round,pad=0.03", facecolor=color[block], edgecolor="black", alpha=.88)
        ax.add_patch(box)
        effective = 1 / (item.effective_ticks * plan.base_tick)
        label = block.name.replace("_", "\n")
        ax.text(x, y + .07, label, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x, y - .18, f"{block.clock.hz:g}→{effective:g} Hz", ha="center", va="center", fontsize=7)

    for connection in satellite.model.connections:
        if connection.observer:
            continue
        source, target = connection.source.block, connection.target.block
        sx, sy = positions[source.path]
        tx, ty = positions[target.path]
        rad = .10 if connection.feedback else 0.0
        arrow = FancyArrowPatch(
            (sx, sy), (tx, ty), arrowstyle="-|>", mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}", color="#555555", lw=1.0,
            linestyle="--" if connection.feedback else "-", alpha=.7, shrinkA=30, shrinkB=30,
        )
        ax.add_patch(arrow)
    ax.set_title("Hierarchical satellite graph — requested → compiled rate\n(dashed edges are one-step held feedback)", fontsize=14)
    return _save(fig, output, filename)


def _fixed_plot(result: RunResult, title: str, output: Path, filename: str) -> Path:
    time_s = result.array("time")
    attitude = result.array("attitude")
    estimates = result.history["estimate"]
    estimate_q = np.asarray([value.q for value in estimates])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(time_s, _q_error_deg(attitude[:, 3:7]), label="truth")
    axes[0].plot(time_s, _q_error_deg(estimate_q), label="estimate")
    axes[0].set_ylabel("attitude error [deg]")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(time_s, np.rad2deg(np.linalg.norm(attitude[:, :3], axis=1)))
    axes[1].set_ylabel("body rate [deg/s]"); axes[1].grid(alpha=.25)
    names = sorted(result.execution_counts)
    requested = []
    effective = []
    for name in names:
        entries = [entry for entry in result.trace if entry.block == name]
        requested.append(entries[0].requested_hz if entries else 0)
        effective.append(entries[0].effective_hz if entries else 0)
    y = np.arange(len(names))
    axes[2].barh(y+.16, requested, height=.3, label="requested", alpha=.55)
    axes[2].barh(y-.16, effective, height=.3, label="compiled")
    axes[2].set_yticks(y, names, fontsize=8); axes[2].set_xlabel("frequency [Hz]")
    axes[2].legend(); axes[2].grid(axis="x", alpha=.25)
    fig.suptitle(f"{title} — base tick {result.base_tick:g} s, wall {result.wall_time:.3f} s")
    return _save(fig, output, filename)


def fixed_demo(output: Path, *, nested: bool = False):
    # Exclude one-time Numba compilation from the reported steady-state wall time.
    FixedStepEngine().run(build_model(nested_sensors=nested).model, .01, seed=7)
    satellite = build_model(nested_sensors=nested)
    plot_graph(satellite, output, "11_subblock_graph.png" if nested else "10_fixed_graph.png")
    result = FixedStepEngine().run(satellite.model, 12.0, seed=7)
    filename = "12_fixed_subblocks.png" if nested else "10_fixed_step.png"
    title = "Fixed-step with nested sensor subblocks" if nested else "Fixed-step monolithic sensors"
    return result, _fixed_plot(result, title, output, filename)


def async_demo(output: Path):
    FixedStepEngine().run(build_model(nested_sensors=True).model, .01, seed=7)
    satellite = build_model(nested_sensors=True)
    result = AsyncEventEngine().run(satellite.model, 3.0, seed=7)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    time_s = result.array("time")
    attitude = result.array("attitude")
    axes[0].plot(time_s, _q_error_deg(attitude[:, 3:7]))
    axes[0].set_ylabel("attitude error [deg]"); axes[0].grid(alpha=.25)
    names = sorted(result.execution_counts)
    row = {name: i for i, name in enumerate(names)}
    for entry in result.trace:
        if entry.time <= .5:
            axes[1].scatter(entry.time, row[entry.block], marker="|", s=65)
    axes[1].set_yticks(range(len(names)), names, fontsize=8)
    axes[1].set_xlim(0, .5); axes[1].set_xlabel("simulation time [s]")
    axes[1].set_title("Independent requested clocks; zero communication delay")
    axes[1].grid(axis="x", alpha=.25)
    fig.suptitle("Asynchronous execution of the same nested model")
    return result, _save(fig, output, "13_asynchronous_zero_delay.png")


def _mc_plot(result: BatchResult, output: Path, filename: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for run in range(min(3, result.attitude.shape[1])):
        axes[0].plot(result.time, _q_error_deg(result.attitude[:, run, 3:7]), label=f"run {run}")
        axes[1].plot(result.time, np.rad2deg(np.linalg.norm(result.attitude[:, run, :3], axis=1)))
    axes[0].set_ylabel("attitude error [deg]"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].set_ylabel("body rate [deg/s]"); axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=.25)
    fig.suptitle(
        f"{result.backend} on {result.device}\n"
        f"compile {result.compile_time:.3f} s, execute {result.wall_time:.4f} s, "
        f"{result.satellite_steps_per_second/1e6:.2f} M satellite-steps/s"
    )
    return _save(fig, output, filename)


def cpu_mc_demo(output: Path):
    result = run_numba_monte_carlo(3, 12.0)
    return result, _mc_plot(result, output, "14_cpu_numba_mc.png")


def gpu_mc_demo(output: Path, allow_cpu: bool = False):
    result = run_jax_monte_carlo(3, 12.0, require_gpu=not allow_cpu)
    return result, _mc_plot(result, output, "15_gpu_jax_mc.png")


def power_demo(output: Path):
    FixedStepEngine().run(build_model(nested_sensors=True, with_power=True).model, .01, seed=7)
    satellite = build_model(nested_sensors=True, with_power=True)
    plot_graph(satellite, output, "16_power_graph.png")
    result = FixedStepEngine().run(satellite.model, 120.0, seed=7)
    time_s = result.array("time")
    power: list[PowerState] = result.history["power"]
    actuation = result.history["actuation"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(time_s, [p.generated_w for p in power], label="solar generation")
    axes[0].plot(time_s, [p.load_w for p in power], label="load")
    axes[0].set_ylabel("power [W]"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(time_s, [100*p.soc for p in power], label="battery SoC")
    axes[1].plot(time_s, [10*p.voltage for p in power], label="voltage × 10")
    axes[1].set_ylabel("percent / scaled volts"); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].plot(time_s, [np.linalg.norm(a.body_torque) for a in actuation])
    axes[2].set_ylabel("actuator torque [N m]"); axes[2].set_xlabel("time [s]"); axes[2].grid(alpha=.25)
    fig.suptitle("Power added as ordinary nested blocks: attitude → panels → battery → actuators")
    return result, _save(fig, output, "17_power_results.png")


def benchmark_demo(output: Path, allow_jax_cpu: bool = False):
    # Warm the imported attitude kernel before timing the Python graph.
    warm = build_model(); FixedStepEngine().run(warm.model, .02)
    graph = build_model()
    interpreted = FixedStepEngine().run(graph.model, 20.0, seed=3)
    # A three-run demonstration is intentionally too small to benefit from a
    # GPU. Use a moderate batch here to measure throughput rather than launch
    # overhead, while remaining practical on student hardware.
    benchmark_runs, benchmark_duration = 1024, 10.0
    cpu = run_numba_monte_carlo(benchmark_runs, benchmark_duration)
    gpu = run_jax_monte_carlo(benchmark_runs, benchmark_duration, require_gpu=not allow_jax_cpu)
    realtime = 20.0 / interpreted.wall_time
    throughput = [cpu.satellite_steps_per_second, gpu.satellite_steps_per_second]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    bar = axes[0].bar(["Python block graph"], [realtime], color="#5b8ff9")[0]
    axes[0].text(bar.get_x()+bar.get_width()/2, realtime, f"{realtime:,.0f}×", ha="center", va="bottom")
    axes[0].set_ylabel("simulated time / wall time"); axes[0].grid(axis="y", alpha=.25)
    bars = axes[1].bar(["Numba CPU", "JAX GPU"], throughput, color=["#61dDAA", "#65789b"])
    for bar, value in zip(bars, throughput):
        axes[1].text(bar.get_x()+bar.get_width()/2, value, f"{value/1e6:.2f} M", ha="center", va="bottom")
    axes[1].set_ylabel("satellite-steps/s"); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle(
        f"Execution benchmark: {benchmark_runs} × {benchmark_duration:g}s (steady-state only)\nNumba compile {cpu.compile_time:.2f}s; "
        f"JAX compile {gpu.compile_time:.2f}s; GPU={gpu.device}"
    )
    print(f"Python graph: {realtime:.1f}× real time")
    print(f"Numba CPU:   {throughput[0]/1e6:.3f} M satellite-steps/s")
    print(f"JAX {gpu.device}: {throughput[1]/1e6:.3f} M satellite-steps/s")
    return {"python": interpreted, "numba": cpu, "jax": gpu}, _save(fig, output, "18_benchmark.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", nargs="?", default="all", choices=("fixed", "subblocks", "async", "cpu-mc", "gpu-mc", "power", "benchmark", "all"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output_v2"))
    parser.add_argument("--allow-jax-cpu", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    jobs = {
        "fixed": lambda: fixed_demo(args.output),
        "subblocks": lambda: fixed_demo(args.output, nested=True),
        "async": lambda: async_demo(args.output),
        "cpu-mc": lambda: cpu_mc_demo(args.output),
        "gpu-mc": lambda: gpu_mc_demo(args.output, args.allow_jax_cpu),
        "power": lambda: power_demo(args.output),
        "benchmark": lambda: benchmark_demo(args.output, args.allow_jax_cpu),
    }
    selected = jobs if args.demo == "all" else {args.demo: jobs[args.demo]}
    failures = 0
    for name, job in selected.items():
        try:
            _, path = job()
            print(f"{name:10s} -> {path}")
        except RuntimeError as exc:
            failures += 1
            print(f"{name:10s} -> unavailable: {exc}")
    if args.show:
        plt.show()
    else:
        plt.close("all")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
