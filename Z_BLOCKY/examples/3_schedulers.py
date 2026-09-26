"""Compare Base, optimized Asynchronous, and Academic scheduling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Z_BLOCKY import (
    AcademicScheduler, AsynchronousScheduler, BaseScheduler, Block,
    Environment, GaussianNoise, Plant, Port, QueriedBlock, QueryPort,
    Simulation,
)


class Gyroscope(Block):
    """Stateless sampled model; skipped samples do not advance hidden state."""

    def __init__(self) -> None:
        super().__init__(
            "gyroscope",
            frequency_hz=20.0,
            execution_s=0.010,
            delay_s=0.005,
            delay_noise=GaussianNoise(mean=0.020, standard_deviation=0.015),
            sample_on_demand=True,
            outputs=(Port("gyro_rate", float),),
        )

    def step(self, inputs, context=None):
        return {"gyro_rate": 0.2 + 0.01 * context.time_s}


class Disturbance(QueriedBlock):
    def __init__(self) -> None:
        super().__init__(
            "disturbance",
            queries=(QueryPort("torque", dict, float),),
            execution_s=0.002,
            delay_s=0.001,
        )

    def query(self, port, request, context=None):
        return 0.001 * request["rate"]


class Controller(Block):
    def __init__(self) -> None:
        super().__init__(
            "controller",
            frequency_hz=1.0,
            execution_s=0.100,
            delay_s=0.010,
            inputs=(Port("gyro_rate", float),),
            outputs=(Port("command", float),),
        )
        self.observations: list[tuple[float, float]] = []

    def step(self, inputs, context=None):
        rate = inputs["gyro_rate"]
        torque = context.query("disturbance", "torque", {"rate": rate})
        self.observations.append((context.time_s, rate))
        return {"command": -rate - torque}


def build_case():
    environment = Environment(disturbance=Disturbance())
    plant = Plant("spacecraft")
    plant.add(Gyroscope())
    controller = plant.add(Controller())
    plant.auto_connect()
    plant.query(controller, environment["disturbance"], "torque")
    return Simulation(environment=environment, plants=(plant,)), controller


def show_result(label, result, controller):
    print(f"\n{label}")
    print(f"  gyro executions: {result.execution_counts['spacecraft.gyroscope']}")
    print(f"  nominal gyro releases coalesced/skipped: {result.skipped_ticks['spacecraft.gyroscope']}")
    print(f"  controller executions: {result.execution_counts['spacecraft.controller']}")
    print(f"  controller skipped for missing first sample: {result.skipped_ticks['spacecraft.controller']}")
    print(f"  controller's latest completed samples: {controller.observations}")
    print(f"  wall time: {result.wall_time_s * 1e3:.3f} ms")


def plot_timelines(cases, duration_s: float, *, show: bool = False) -> tuple[Path, Path]:
    """Plot nominal releases in gray and completed block evaluations in green."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
    gray = "#cbd5e1"
    green = "#16a34a"
    text = "#1e293b"

    for ax, (title, simulation, result) in zip(axes, cases):
        blocks = simulation.blocks
        row_count = len(blocks)
        labels = []
        if result.scheduler == "academic":
            nominal_rate = min(block.frequency_hz for block in blocks)
        else:
            nominal_rate = None

        for row, block in enumerate(blocks):
            y = row_count - row - 1
            path = f"{block._owner.removeprefix('plant.')}.{block.name}"
            labels.append(path)
            ax.hlines(y, 0, duration_s, color="#e2e8f0", linewidth=1.0, zorder=0)

            release_rate = nominal_rate or block.frequency_hz
            period = 1.0 / release_rate
            count = int(duration_s / period + 1e-10) + 1
            releases = [tick * period for tick in range(count) if tick * period <= duration_s + 1e-10]
            if releases:
                ax.vlines(releases, y - 0.13, y + 0.13, color=gray, linewidth=1.0, zorder=1)

            starts = [event.time_s for event in result.executions if event.block == path]
            for start in starts:
                # Show tiny green marks for instantaneous academic evaluations.
                width = block.execution_s if result.scheduler != "academic" else 0.0
                width = max(width, period * 0.035, 0.008)
                width = min(width, max(0.0, duration_s - start))
                if width > 0:
                    ax.broken_barh([(start, width)], (y - 0.19, 0.38), facecolors=green, edgecolors="#15803d", linewidth=0.6, zorder=3)
                else:
                    ax.scatter([start], [y], marker="|", s=110, color=green, linewidths=2.0, zorder=3)

        ax.set_yticks(range(row_count), labels[::-1])
        ax.set_ylim(-0.65, row_count - 0.35)
        ax.set_title(title, loc="left", color=text, fontsize=11, fontweight="semibold", pad=8)
        ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        ax.tick_params(axis="y", length=0, labelcolor=text, labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")

    axes[-1].set_xlim(0, duration_s)
    axes[-1].set_xlabel("Simulation time (s)", color=text)
    axes[-1].set_xticks([i * duration_s / 10 for i in range(11)])
    fig.suptitle("Block scheduling timelines", fontsize=16, fontweight="semibold", color=text)
    fig.legend(
        handles=[
            Patch(facecolor=gray, edgecolor=gray, label="Nominal release opportunity"),
            Patch(facecolor=green, edgecolor="#15803d", label="Block evaluated"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False,
    )
    output_base = Path(__file__).with_name("3_scheduler_timelines")
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the timeline plot after saving it")
    args = parser.parse_args()

    duration = 2.0
    cases = []
    simulation, controller = build_case()
    result = BaseScheduler().run(simulation, duration, seed=9)
    show_result("Base: independent 20 Hz gyro and 1 Hz controller", result, controller)
    cases.append(("Base · independent block rates", simulation, result))

    simulation, controller = build_case()
    result = AsynchronousScheduler().run(simulation, duration, seed=9)
    show_result("Asynchronous: capture every gyro tick; evaluate only used samples", result, controller)
    print("  Late noisy samples remain unpublished until completion; the controller uses the previous value.")
    cases.append(("Asynchronous · deferred gyro evaluation", simulation, result))

    simulation, controller = build_case()
    result = AcademicScheduler().run(simulation, duration, seed=9)
    show_result("Academic: both blocks at the plant's slowest rate; zero modeled latency", result, controller)
    cases.append(("Academic · common slowest rate", simulation, result))

    png_path, svg_path = plot_timelines(cases, duration, show=args.show)
    print(f"\nTimeline plot saved to:\n  {png_path}\n  {svg_path}")


if __name__ == "__main__":
    main()
