"""Render the v3 user-facing API demonstrations.

Run with ``venv/bin/python -m debug.debug_3_0.run_v3 all``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .block_engine import compile_fixed
from .user_api import (
    Environment,
    InitialState,
    Integration,
    Scenario,
    Satellite,
    Simulation,
)


USER_CODE = '''\
from debug.debug_3_0.user_api import (
    Environment, InitialState, Integration,
    Scenario, Satellite, Simulation,
)

# Satellite owns its hardware, flight software, and initial state.
satellite = Satellite.standard_cube_sat(
    name="solo",
    sensors=False,
    estimator=False,     # controller receives ideal state knowledge
    initial_state=InitialState.leo_default(),
)

environment = Environment.standard_leo(
    gravity="J2",
    magnetic_field="dipole",
    atmosphere="exponential",
)

# Scenario owns numerical and execution policy.
scenario = Scenario(
    duration=60.0,
    kernel="Numba",     # "Python", "Numba", or "JAX"
    run="Single",       # "Single" or "Monte Carlo"
    integration=Integration(
        orbit_hz=1.0,
        attitude_hz=1.0,
        subsystem_hz=1.0,
    ),
)

simulation = Simulation(
    satellites={"solo": satellite},
    environment=environment,
    scenario=scenario,
)

result = simulation.run()
'''


def _save(fig, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.savefig(path, dpi=170, bbox_inches="tight")
    return path


def _box(ax, x, y, title, subtitle="", *, width=1.7, height=.72, color="#6aaed6"):
    patch = FancyBboxPatch(
        (x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.04", facecolor=color, edgecolor="#22313f", alpha=.9,
    )
    ax.add_patch(patch)
    ax.text(x, y+.09, title, ha="center", va="center", fontsize=8.8, weight="bold")
    if subtitle:
        ax.text(x, y-.17, subtitle, ha="center", va="center", fontsize=7.4)


def _arrow(ax, source, target, *, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        source, target, arrowstyle="-|>", mutation_scale=10, color="#4c5866",
        lw=1.05, linestyle="--" if dashed else "-", alpha=.8,
        connectionstyle=f"arc3,rad={rad}", shrinkA=24, shrinkB=24,
    ))


def plot_single_graph(simulation: Simulation, output: Path, filename: str) -> Path:
    compiled = simulation.compile()
    model = compiled.model
    plan = compile_fixed(model)
    leaves = [item.block for item in plan.blocks]
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.set(xlim=(0, 13), ylim=(0, 6.2)); ax.axis("off")
    ax.add_patch(Rectangle((2.15, .55), 9.7, 5.15, facecolor="#edf5ff", edgecolor="#3478b9", lw=1.8))
    ax.text(2.4, 5.42, "Satellite: actual compiled leaf blocks", color="#24547d", weight="bold")
    positions = {}
    colors = {"environment": "#9ecae1", "recorder": "#d9d9d9", "orbit": "#80b1d3", "attitude": "#80b1d3", "disturbances": "#b3de69", "sensors": "#ccebc5", "estimator": "#b3de69", "state_knowledge": "#ccebc5", "controller": "#fbb4ae", "actuators": "#fbb4ae"}
    layout = {
        "environment_models": (1.05, 3.2), "orbit_propagation": (3.25, 4.35),
        "attitude_propagation": (3.25, 2.05), "disturbances": (5.25, 3.2),
        "sensors": (7.15, 4.35), "estimator": (7.15, 2.05),
        "state_knowledge": (7.15, 2.05), "controller": (9.15, 3.2),
        "actuators": (10.95, 4.35), "recorder": (10.95, 1.15),
    }
    for index, block in enumerate(leaves):
        x, y = layout.get(block.name, (2.5 + index, 3.2))
        positions[block] = (x, y)
        leaf = block.name.replace("_", " ")
        hz = f"{block.clock.hz:g} Hz"
        color = colors.get(block.name, "#fdb863")
        _box(ax, x, y, leaf, hz, width=1.45, height=.78, color=color)
    for connection in model.connections:
        if connection.source.block not in positions or connection.target.block not in positions:
            continue
        sx, sy = positions[connection.source.block]; tx, ty = positions[connection.target.block]
        _arrow(ax, (sx, sy-.40), (tx, ty+.40), dashed=connection.feedback, rad=.10 if sx != tx else .0)
    ax.text(.35, .72, "solid = same-step data dependency   dashed = held feedback dependency", fontsize=8, color="#4c5866")
    ax.set_title("Actual compiled graph: every plotted block is lowered into the Numba/JAX fixed-step kernel", fontsize=14)
    return _save(fig, output, filename)


def plot_formation_graph(simulation: Simulation, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.set(xlim=(0, 13), ylim=(0, 5.4)); ax.axis("off")
    _box(ax, 1.0, 4.5, "Environment", "shared gravity / field / stations", width=2.0, color="#9ecae1")
    for x, name, color in ((3.9, "chief", "#edf5ff"), (7.4, "deputy", "#fff0e6")):
        ax.add_patch(Rectangle((x-1.45, .7), 2.9, 3.65, facecolor=color, edgecolor="#3478b9", lw=1.7))
        ax.text(x-1.3, 4.14, f"Satellite: {name}", weight="bold", color="#24547d")
        _box(ax, x, 3.25, "State + propagation", "initial orbit/attitude\nindependent runtime state", width=2.1, color="#80b1d3")
        _box(ax, x, 2.15, "GPS + state knowledge", "GPS position estimate", width=1.7, color="#b3de69")
        _box(ax, x, 1.15, "ADCS", "controller + actuators", width=1.65, color="#fbb4ae")
        _arrow(ax, (x, 3.25), (x, 2.15)); _arrow(ax, (x, 3.25), (x, 1.15))
    _box(ax, 10.7, 2.7, "Formation controller", "r_deputy − r_chief", width=2.2, color="#bebada")
    _box(ax, 12.3, 1.0, "Recorder", "relative position", width=1.3, color="#d9d9d9")
    _arrow(ax, (3.9, 2.15), (10.7, 2.7), rad=.12)
    _arrow(ax, (7.4, 2.15), (10.7, 2.7), rad=-.08)
    _arrow(ax, (10.7, 2.7), (12.3, 1.0))
    ax.set_title("Outer layer: shared Environment, N Satellites, and cross-satellite formation logic", fontsize=14)
    return _save(fig, output, "04_two_satellite_formation_graph.png")


def code_demo(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "00_user_facing_example.py").write_text(USER_CODE)
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.axis("off")
    ax.text(.01, .99, USER_CODE, va="top", family="monospace", fontsize=9.5)
    ax.set_title("User-facing v3 API", loc="left", fontsize=15, weight="bold")
    return _save(fig, output, "00_user_facing_code.png")


def _single_numba_simulation() -> Simulation:
    satellite = Satellite.standard_cube_sat(
        "solo", sensors=True, estimator=True, initial_state=InitialState.leo_default(),
    )
    return Simulation(
        satellites={"solo": satellite},
        environment=Environment.standard_leo(),
        scenario=Scenario(
            duration=30.0,
            kernel="Numba",
            run="Single",
            integration=Integration(orbit_hz=1.0, attitude_hz=1.0, subsystem_hz=1.0),
        ),
    )


def numba_single_demo(output: Path) -> Path:
    simulation = _single_numba_simulation()
    plot_single_graph(simulation, output, "01_single_numba_graph.png")
    result = simulation.run()
    attitude = result.attitude[:, 0]
    q = attitude[:, 3:7]
    error = np.rad2deg(2 * np.arctan2(np.linalg.norm(q[:, 1:], axis=1), np.abs(q[:, 0])))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
    axes[0].plot(result.time, error); axes[0].set_ylabel("attitude error [deg]"); axes[0].grid(alpha=.25)
    axes[1].plot(result.time, np.rad2deg(np.linalg.norm(attitude[:, :3], axis=1)))
    axes[1].set_ylabel("body rate [deg/s]"); axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=.25)
    fig.suptitle(
        f"Numba, Single — complete orbit/sensors/estimator/ADCS graph; 1 Hz schedule\n"
        f"compile {result.compile_time:.3f}s; execute {result.wall_time:.5f}s"
    )
    return _save(fig, output, "01_single_numba_results.png")


def state_knowledge_demo(output: Path) -> Path:
    satellite = Satellite.standard_cube_sat(
        "solo", sensors=False, estimator=False, initial_state=InitialState.leo_default(),
    )
    simulation = Simulation(
        satellites={"solo": satellite}, environment=Environment.standard_leo(),
        scenario=Scenario(
            duration=20.0, kernel="Python", run="Single",
            integration=Integration(orbit_hz=1.0, attitude_hz=1.0, subsystem_hz=1.0),
        ),
    )
    result = simulation.run()
    attitude = np.asarray(result.history["attitude"])
    estimate = np.asarray([value.q for value in result.history["estimate"]])
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(result.history["time"], np.rad2deg(2*np.arctan2(np.linalg.norm(attitude[:, 4:7], axis=1), np.abs(attitude[:, 3]))), label="truth")
    ax.plot(result.history["time"], np.rad2deg(2*np.arctan2(np.linalg.norm(estimate[:, 1:], axis=1), np.abs(estimate[:, 0]))), "--", label="state knowledge")
    ax.set(xlabel="time [s]", ylabel="attitude error [deg]", title="No sensor or estimator blocks — controller receives ideal state knowledge")
    ax.grid(alpha=.25); ax.legend()
    return _save(fig, output, "02_state_knowledge_no_sensors.png")


def formation_demo(output: Path) -> Path:
    chief_state = InitialState.leo_default()
    deputy_orbit = chief_state.orbit.copy(); deputy_orbit[:3] += np.array([100.0, 0.0, 0.0])
    deputy_state = InitialState(deputy_orbit, chief_state.attitude.copy())
    simulation = Simulation(
        satellites={
            "chief": Satellite.standard_cube_sat("chief", sensors=False, estimator=False, initial_state=chief_state),
            "deputy": Satellite.standard_cube_sat("deputy", sensors=False, estimator=False, initial_state=deputy_state),
        },
        environment=Environment.standard_leo(ground_stations=("Svalbard",)),
        scenario=Scenario(
            duration=90.0, kernel="Python", run="Single",
            integration=Integration(orbit_hz=1.0, attitude_hz=1.0, subsystem_hz=1.0),
        ),
    )
    plot_formation_graph(simulation, output)
    result = simulation.run()
    relative = np.asarray(result.history["relative_position_m"])
    time = np.asarray(result.history["time"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(time, relative[:, 0], label="Δx")
    axes[0].plot(time, relative[:, 1], label="Δy")
    axes[0].plot(time, relative[:, 2], label="Δz")
    axes[0].set_ylabel("GPS relative position [m]"); axes[0].grid(alpha=.25); axes[0].legend(ncol=3)
    axes[1].plot(time, np.linalg.norm(relative, axis=1))
    axes[1].set(xlabel="time [s]", ylabel="separation [m]"); axes[1].grid(alpha=.25)
    fig.suptitle("Formation controller consumes the GPS state estimate from each satellite")
    return _save(fig, output, "05_two_satellite_formation_results.png")


def _benchmark_simulation(kernel: str, run: str, monte_carlo_runs: int) -> Simulation:
    satellite = Satellite.standard_cube_sat(
        "solo", sensors=True, estimator=True, initial_state=InitialState.leo_default(),
    )
    return Simulation(
        satellites={"solo": satellite}, environment=Environment.standard_leo(),
        scenario=Scenario(
            duration=20.0, kernel=kernel, run=run, monte_carlo_runs=monte_carlo_runs,
            integration=Integration(orbit_hz=10.0, attitude_hz=10.0, subsystem_hz=10.0),
        ),
    )


def benchmark_demo(output: Path) -> Path:
    """Measure the public API's single and Monte-Carlo selections."""
    rows = []
    # Warm the Numba attitude primitive used by the full Python graph.
    _benchmark_simulation("Python", "Single", 1).run()
    for kernel in ("Python", "Numba", "JAX"):
        for run, count in (("Single", 1), ("Monte Carlo", 64)):
            try:
                result = _benchmark_simulation(kernel, run, count).run()
                if kernel == "Python":
                    results = result if isinstance(result, list) else [result]
                    wall = sum(item.wall_time for item in results)
                    compile_time = 0.0
                    detail = "full Python graph"
                else:
                    wall = result.wall_time
                    compile_time = result.compile_time
                    detail = result.device
                rows.append((kernel, run, count, wall, compile_time, detail))
            except RuntimeError as exc:
                rows.append((kernel, run, count, np.nan, np.nan, str(exc)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    kernels = ["Python", "Numba", "JAX"]
    for index, run in enumerate(("Single", "Monte Carlo")):
        subset = [row for row in rows if row[1] == run]
        values = [next(row[3] for row in subset if row[0] == kernel) for kernel in kernels]
        axes[index].bar(kernels, values, color=["#5b8ff9", "#61dDAA", "#65789b"])
        axes[index].set_yscale("log")
        axes[index].set_ylabel("steady-state wall time [s]")
        axes[index].set_title(f"{run}: {subset[0][2]} satellite run(s)")
        axes[index].grid(axis="y", alpha=.25)
        for x, value in enumerate(values):
            if np.isfinite(value):
                axes[index].text(x, value, f"{value:.3g}s", ha="center", va="bottom")
    fig.suptitle(
        "Kernel × run-mode benchmark — 20 s at 10 Hz; compilation reported separately\n"
        "All three backends now execute the complete compiled graph: orbit, disturbances, sensors/estimator, controller, actuators, attitude."
    )
    path = _save(fig, output, "06_kernel_run_benchmark.png")
    lines = ["kernel\trun\truns\tsteady_s\tcompile_s\tdetail"]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    (output / "06_kernel_run_benchmark.tsv").write_text("\n".join(lines) + "\n")
    for row in rows:
        print(f"{row[0]:6s} | {row[1]:11s} | n={row[2]:2d} | steady={row[3]:.5f}s | compile={row[4]:.3f}s | {row[5]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", nargs="?", default="all", choices=("code", "numba", "state", "formation", "benchmark", "all"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output_v3"))
    args = parser.parse_args()
    jobs = {
        "code": lambda: code_demo(args.output),
        "numba": lambda: numba_single_demo(args.output),
        "state": lambda: state_knowledge_demo(args.output),
        "formation": lambda: formation_demo(args.output),
        "benchmark": lambda: benchmark_demo(args.output),
    }
    selected = jobs if args.demo == "all" else {args.demo: jobs[args.demo]}
    for name, job in selected.items():
        print(f"{name:10s} -> {job()}")
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
