"""Reusable timing diagram for a complete simulation run."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocky import Simulation
    from .schedulers import SchedulerResult


def plot_timeline(
    simulation: Simulation,
    result: SchedulerResult,
    *,
    output: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Show all plants, nominal ticks, sample completions, and real evaluations.

    With ``output`` as a path stem, saves both PNG and SVG and returns their
    paths. Without it, returns the Matplotlib figure for further customization.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = []
    for plant in simulation.plants:
        if plant.dynamics is not None:
            rows.append((plant.name, None))
        rows.extend((plant.name, block) for block in plant.blocks)
    if not rows:
        raise ValueError("the simulation has no dynamics or blocks to plot")

    fig, ax = plt.subplots(figsize=(12, max(3.8, 1.7 + 0.62 * len(rows))))
    gray, green, purple, ink = "#94a3b8", "#16a34a", "#8b5cf6", "#1e293b"
    captures = defaultdict(list)
    executions = defaultdict(list)
    for record in result.captures:
        captures[record.block].append(record)
    for record in result.executions:
        executions[record.block].append(record)

    y_positions = []
    labels = []
    previous_plant = None
    for row_index, (plant_name, block) in enumerate(rows):
        y = len(rows) - row_index - 1
        y_positions.append(y)
        if previous_plant is not None and plant_name != previous_plant:
            ax.axhline(y + 0.5, color="#cbd5e1", linewidth=1.2)
        previous_plant = plant_name
        if block is None:
            labels.append(f"{plant_name}  ·  dynamics")
            ax.hlines(y, 0, result.final_time_s, color=purple, linewidth=6, alpha=0.75, zorder=2)
            continue

        path = f"{plant_name}.{block.name}"
        labels.append(f"{plant_name}  ·  {block.name}")
        ax.hlines(y, 0, result.final_time_s, color="#e2e8f0", linewidth=1, zorder=0)
        period = 1.0 / block.frequency_hz
        if result.scheduler == "academic":
            plant = next(plant for plant in simulation.plants if plant.name == plant_name)
            period = max(1.0 / item.frequency_hz for item in plant.blocks)

        if captures[path]:
            used_samples = {
                event.sampled_at_s for event in executions[path]
                if event.sampled_at_s is not None
            }
            represented_ticks = {capture.sampled_at_s for capture in captures[path]}
            for capture in captures[path]:
                color = green if capture.sampled_at_s in used_samples else gray
                end = min(capture.completed_at_s, result.final_time_s)
                start = capture.sampled_at_s
                ax.plot([start, end], [y + 0.18, y - 0.18], color=color,
                        linewidth=1.5, solid_capstyle="round", zorder=3)
                if capture.completed_at_s > result.final_time_s:
                    ax.scatter(result.final_time_s, y - 0.18, marker=">", s=16,
                               color=color, zorder=4)
            count = int(result.final_time_s / period + 1e-10) + 1
            for tick in (i * period for i in range(count)):
                if tick <= result.final_time_s + 1e-10 and tick not in represented_ticks:
                    ax.vlines(tick, y - 0.12, y + 0.12, color=gray, linewidth=0.85, zorder=1)
        else:
            count = int(result.final_time_s / period + 1e-10) + 1
            ticks = [i * period for i in range(count) if i * period <= result.final_time_s + 1e-10]
            executed_ticks = {event.time_s for event in executions[path]}
            skipped_ticks = [tick for tick in ticks if tick not in executed_ticks]
            if skipped_ticks:
                ax.vlines(skipped_ticks, y - 0.12, y + 0.12, color=gray, linewidth=0.85, zorder=1)
            for event in executions[path]:
                end_at = event.completed_at_s if event.completed_at_s is not None else event.time_s
                end = min(end_at, result.final_time_s)
                ax.plot([event.time_s, end], [y + 0.18, y - 0.18], color=green,
                        linewidth=1.7, solid_capstyle="round", zorder=4)
                if end_at > result.final_time_s:
                    ax.scatter(result.final_time_s, y - 0.18, marker=">", s=16,
                               color=green, zorder=4)

    ax.set_yticks(y_positions, labels)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    padding = max(0.025, result.final_time_s * 0.015)
    ax.set_xlim(-padding, result.final_time_s + padding)
    ax.set_xlabel("Simulation time (s)", color=ink)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.75)
    ax.tick_params(axis="y", length=0, labelcolor=ink, labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.suptitle(title or f"{result.scheduler.title()} scheduler · simulation timeline",
                 color=ink, fontsize=15, fontweight="semibold", y=0.98)
    fig.legend(handles=[
        Line2D([0], [0], color=purple, lw=5, label="Continuous integration"),
        Line2D([0], [0], color=gray, lw=1.5, label="Triggered; not used"),
        Line2D([0], [0], color=green, lw=1.7, label="Executed / sample used"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.14, top=0.76)

    if output is None:
        if show:
            plt.show()
        return fig
    stem = Path(output)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return png, svg
