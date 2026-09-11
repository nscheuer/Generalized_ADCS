"""Plot mean +/- 1-sigma attitude convergence for the latest 6U MC results."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ADCS
from plot_style import configure_ieee_style


CAMPAIGN_PREFIX = "6u_estimator_nodisturbance"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONFIGURATIONS = ((0, "qp", "3+0"), (1, "lp", "3+1"), (3, "lp", "3+3"))


def _latest_sim(output_dir: Path, number_rw: int, allocator: str) -> Path:
    pattern = f"{CAMPAIGN_PREFIX}_3mtq_{number_rw}rw_{allocator}_mc_*.sim"
    candidates = list(output_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No saved MC result matching {pattern!r} in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _angle_error_deg(run) -> tuple[np.ndarray, np.ndarray]:
    """Return time and true-to-target quaternion angle error in degrees."""
    time = np.asarray(run.time_s, dtype=float)
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.sum(states * targets, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return time, np.rad2deg(2.0 * np.arccos(dots))


def _summary(results: ADCS.SimulationResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traces = [_angle_error_deg(run) for run in results.runs]
    time = traces[0][0]
    values = np.vstack([np.interp(time, run_time, angle) for run_time, angle in traces])
    return time, np.mean(values, axis=0), np.std(values, axis=0)


def _final_error_summary(results: ADCS.SimulationResults) -> tuple[float, float, float]:
    """Return final-error mean, 1-sigma magnitude, and maximum final error."""
    final_errors = []
    for run in results.runs:
        _, angle = _angle_error_deg(run)
        final_errors.append(angle[-1])
    final_errors = np.asarray(final_errors)
    return (float(np.mean(final_errors)), float(np.std(final_errors)),
            float(np.max(final_errors)))


def _plot_convergence_scatter(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    colors = ("#0072B2", "#D55E00", "#009E73")
    points = []
    for (number_rw, allocator, label), color in zip(CONFIGURATIONS, colors):
        result_path = _latest_sim(output_dir, number_rw, allocator)
        results = ADCS.SimulationResults.load(result_path)
        final_error, sigma, _ = _final_error_summary(results)
        ax.scatter(sigma, final_error, s=34, color=color, zorder=3)
        points.append((sigma, final_error, label, color))

    for index, (x, y, label, color) in enumerate(points):
        ax.annotate(label, (x, y), xytext=(5, 5 + 14 * index),
                    textcoords="offset points", color=color)

    ax.set_xlabel("Final pointing error, 1σ [deg]")
    ax.set_ylabel("Final mean pointing error [deg]")
    ax.set_title("Pointing-error mean vs. 1σ magnitude")
    ax.grid(True)
    ax.set_ylim(bottom=0.0)
    xmax = max(x for x, _, _, _ in points)
    ax.set_xlim(left=0.0, right=max(1.0, xmax * 1.15))
    fig.tight_layout()
    path = output_dir / f"{CAMPAIGN_PREFIX}_pointing_error_scatter_mc.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    configure_ieee_style()
    output_dir = OUTPUT_DIR
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    colors = ("#0072B2", "#D55E00", "#009E73")

    for (number_rw, allocator, label), color in zip(CONFIGURATIONS, colors):
        result_path = _latest_sim(output_dir, number_rw, allocator)
        results = ADCS.SimulationResults.load(result_path)
        time, mean, sigma = _summary(results)
        ax.plot(time / 60.0, mean, color=color, label=label)
        ax.fill_between(time / 60.0, np.maximum(mean - sigma, 0.0), mean + sigma,
                        color=color, alpha=0.18, linewidth=0.0)
        print(f"{label}: {result_path.name}")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Angle error [deg]")
    ax.set_title("6U attitude convergence")
    ax.grid(True)
    ax.set_ylim(bottom=0.0)
    ax.legend(title="Configuration", ncol=3, loc="upper right", frameon=True)
    fig.tight_layout()
    path = output_dir / f"{CAMPAIGN_PREFIX}_angle_convergence_mc.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    _plot_convergence_scatter(output_dir)


if __name__ == "__main__":
    main()
