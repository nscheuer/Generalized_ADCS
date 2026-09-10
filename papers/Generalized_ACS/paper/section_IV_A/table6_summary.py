"""Summarize the Section IV-A Monte Carlo ``.sim`` files as a paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ADCS
from ADCS.helpers.math_helpers import quat_inv, quat_mult, rot_mat


CONFIGS = ("3mtq_0rw", "3mtq_1rw", "3mtq_3rw")
TASKS = ("vector",)
ALLOCATORS = ("lp", "qp")
DISPLAY_CONFIG = {"3mtq_0rw": "3MTQ+0RW", "3mtq_1rw": "3MTQ+1RW", "3mtq_3rw": "3MTQ+3RW"}
BODY_BORESIGHT = np.array([0.0, 0.0, 1.0])
THRESHOLD_DEG = 5.0


def final_error_deg(run, task: str) -> float:
    state = run.state_hist[-1]
    q = np.asarray(state.q, dtype=float)
    target = np.asarray(run.target_hist[-1], dtype=float)

    if task == "full":
        q_error = quat_mult(quat_inv(target), q)
        return float(2.0 * np.arccos(np.clip(abs(q_error[0]), 0.0, 1.0))
                     * 180.0 / np.pi)

    goal_vector = target[1:]
    boresight = rot_mat(q) @ BODY_BORESIGHT
    dot = np.clip(np.dot(boresight, goal_vector), -1.0, 1.0)
    return float(np.arccos(dot) * 180.0 / np.pi)


def newest_file(output_dir: Path, config: str, task: str, allocator: str) -> Path | None:
    matches = sorted(output_dir.glob(f"table6_{config}_{task}_{allocator}_*.sim"))
    return matches[-1] if matches else None


def load_errors(path: Path, task: str) -> np.ndarray:
    results = ADCS.SimulationResults.load(path, ephem=ADCS.Ephemeris())
    return np.asarray([final_error_deg(run, task) for run in results], dtype=float)


def summarize(path: Path, task: str) -> tuple[float, float, int, np.ndarray]:
    errors = load_errors(path, task)
    return (float(100.0 * np.mean(errors < THRESHOLD_DEG)),
            float(np.mean(errors)), len(errors), errors)


def make_plot(labels: list[str], vector_errors: list[np.ndarray],
              full_errors: list[np.ndarray], output_dir: Path) -> None:
    """Write the categorical LP vector-versus-full performance plot."""
    vector_mean = np.asarray([np.mean(x) for x in vector_errors])
    full_mean = np.asarray([np.mean(x) for x in full_errors])
    # The requested band is the interval from the mean to the 95th percentile.
    vector_p95 = np.asarray([np.percentile(x, 95) for x in vector_errors])
    full_p95 = np.asarray([np.percentile(x, 95) for x in full_errors])

    # Zero final errors are valid, but cannot be represented on a log axis.
    floor = 1e-3
    clamp = lambda x: np.maximum(np.asarray(x, dtype=float), floor)
    x = np.arange(len(labels), dtype=float)
    vector_mean, vector_p95 = map(clamp, (vector_mean, vector_p95))
    full_mean, full_p95 = map(clamp, (full_mean, full_p95))

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.set_yscale("log")
    ax.fill_between(x, vector_mean, vector_p95, color="#2F6DB0", alpha=0.16, linewidth=0)
    ax.fill_between(x, full_mean, full_p95, color="#C44747", alpha=0.16, linewidth=0)
    ax.plot(x, vector_mean, color="#2F6DB0", linewidth=2.4, marker="o",
            markersize=7, markeredgewidth=0.8, markeredgecolor="white", label="Vector pointing mean")
    ax.plot(x, full_mean, color="#C44747", linewidth=2.4, linestyle=":", marker="s",
            markersize=6.5, markeredgewidth=0.8, markeredgecolor="white", label="Full pointing mean")
    ax.axhline(THRESHOLD_DEG, color="#555555", linewidth=1.2, linestyle=":")
    ax.text(x[0] - 0.12, THRESHOLD_DEG * 1.13, "5° convergence threshold",
            color="#555555", fontsize=10, ha="left", va="bottom")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Final pointing error [deg]", fontsize=13)
    ax.set_xlim(-0.35, len(labels) - 0.65)
    ax.grid(which="major", axis="y", color="#B8B8B8", linewidth=0.8, alpha=0.65)
    ax.grid(which="minor", axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="#333333")
    ax.tick_params(axis="x", pad=10)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.legend(handles=[
        Line2D([0], [0], color="#2F6DB0", marker="o", linewidth=2.4, label="Vector pointing mean (LP)"),
        Line2D([0], [0], color="#C44747", marker="s", linestyle=":", linewidth=2.4, label="Full pointing mean (LP)"),
        Patch(facecolor="#777777", alpha=0.16, edgecolor="none", label="Mean–95th percentile band"),
    ], loc="upper right", frameon=True, facecolor="white", edgecolor="#B8B8B8",
       framealpha=0.92, fontsize=10)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.25, top=0.97)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"table6_summary.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()

    print("Config       Task     Conv. % (LP / QP)   Mean (deg) (LP / QP)")
    print("----------------------------------------------------------------")
    missing = []
    plot_labels, vector_plot, full_plot = [], [], []
    for task in TASKS:
        for config in CONFIGS:
            cells = []
            for allocator in ALLOCATORS:
                path = newest_file(args.outputs, config, task, allocator)
                if path is None:
                    missing.append(f"{config}_{task}_{allocator}")
                    cells.append(None)
                else:
                    cells.append(summarize(path, task))

            if any(cell is None for cell in cells):
                print(f"{DISPLAY_CONFIG[config]:<12} {task:<8} missing data")
                continue
            lp, qp = cells
            plot_labels.append(f"{DISPLAY_CONFIG[config]}\n{task}")
            vector_plot.append(lp[3])
            full_path = newest_file(args.outputs, config, "full", "lp")
            if full_path is None:
                missing.append(f"{config}_full_lp")
            else:
                full_plot.append(load_errors(full_path, "full"))
            print(f"{DISPLAY_CONFIG[config]:<12} {task:<8} "
                  f"{lp[0]:.0f} / {qp[0]:.0f}              "
                  f"{lp[1]:.1f} / {qp[1]:.1f}   (n={lp[2]}/{qp[2]})")

    if missing:
        raise SystemExit("Missing cells: " + ", ".join(missing))
    make_plot(plot_labels, vector_plot, full_plot, args.outputs)
    print(f"Wrote {args.outputs / 'table6_summary.png'} and {args.outputs / 'table6_summary.pdf'}")


if __name__ == "__main__":
    main()
