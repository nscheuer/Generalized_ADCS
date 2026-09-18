"""Synthesize 6U wheel-loss degradation and 3+1 capability recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_OUTPUTS = PAPER_DIR / "6U_noestimator_disturbance/outputs"
OPERATIONS_DIR = PAPER_DIR / "BC2_operations"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(OPERATIONS_DIR))

import ADCS
from plot_style import configure_ieee_style
import operations_envelope as operations


POINTING_LIMIT_DEG = 5.0
TAIL_FRACTION = 0.20
SETTLE_DWELL_S = 120.0
WHEEL_MOMENTUM_MAX_NMS = 15.0e-3


@dataclass(frozen=True)
class Architecture:
    label: str
    number_rw: int
    allocator: str
    path: Path


@dataclass(frozen=True)
class Metrics:
    label: str
    time_s: np.ndarray
    errors_deg: np.ndarray
    settle_s: np.ndarray
    settled: np.ndarray
    tail_compliance: np.ndarray
    tail_rms_deg: np.ndarray
    orbit_uptime: np.ndarray
    momentum_headroom: np.ndarray


def _latest(pattern: str, directory: Path) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No result matching {directory / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def architectures() -> list[Architecture]:
    return [
        Architecture("3+3 LP", 3, "LP", _latest("6u_noestimator_disturbance_3mtq_3rw_lp_mc_10_*.sim", BASELINE_OUTPUTS)),
        Architecture("3+2 LP", 2, "LP", _latest("6u_degraded_3mtq_2rw_lp_mc_10_*.sim", OUTPUT_DIR)),
        Architecture("3+1 LP", 1, "LP", _latest("6u_noestimator_disturbance_3mtq_1rw_lp_mc_10_*.sim", BASELINE_OUTPUTS)),
        Architecture("3+0 QP", 0, "QP", _latest("6u_noestimator_disturbance_3mtq_0rw_qp_mc_10_*.sim", BASELINE_OUTPUTS)),
    ]


def _angle_error_deg(run) -> np.ndarray:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    target = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.einsum("ij,ij->i", q, target)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _settle(time_s: np.ndarray, angle_deg: np.ndarray) -> tuple[float, bool]:
    samples = max(1, int(np.ceil(SETTLE_DWELL_S / np.median(np.diff(time_s)))))
    rolling = np.convolve(
        (angle_deg <= POINTING_LIMIT_DEG).astype(int), np.ones(samples, dtype=int), mode="valid"
    )
    indices = np.flatnonzero(rolling == samples)
    return (float(time_s[indices[0]]), True) if indices.size else (float(time_s[-1]), False)


def measure(architecture: Architecture) -> Metrics:
    results = ADCS.SimulationResults.load(architecture.path)
    time_s = np.asarray(results.runs[0].time_s, dtype=float)
    errors = np.asarray([_angle_error_deg(run) for run in results.runs])
    settle = np.asarray([_settle(time_s, error) for error in errors], dtype=object)
    settle_s = np.asarray([row[0] for row in settle], dtype=float)
    settled = np.asarray([row[1] for row in settle], dtype=bool)
    tail = max(1, int(np.ceil(TAIL_FRACTION * errors.shape[1])))
    tail_compliance = np.mean(errors[:, -tail:] <= POINTING_LIMIT_DEG, axis=1)
    tail_rms = np.sqrt(np.mean(errors[:, -tail:] ** 2, axis=1))
    orbit_uptime = np.mean(errors <= POINTING_LIMIT_DEG, axis=1)

    if architecture.number_rw:
        headroom = []
        for run in results.runs:
            momentum = np.asarray([state.h for state in run.state_hist], dtype=float)
            peak_fraction = np.max(np.abs(momentum) / WHEEL_MOMENTUM_MAX_NMS)
            headroom.append(max(0.0, 1.0 - float(peak_fraction)))
        momentum_headroom = np.asarray(headroom)
    else:
        momentum_headroom = np.full(len(results.runs), np.nan)
    return Metrics(
        architecture.label, time_s, errors, settle_s, settled,
        tail_compliance, tail_rms, orbit_uptime, momentum_headroom,
    )


def _bar_with_sigma(ax, x, values, color, *, scale: float = 1.0):
    mean = np.asarray([np.nanmean(value) for value in values]) * scale
    sigma = np.asarray([
        np.nanstd(value, ddof=1) if np.count_nonzero(np.isfinite(value)) > 1 else 0.0
        for value in values
    ]) * scale
    ax.bar(x, mean, color=color, edgecolor="black", linewidth=0.35, zorder=2)
    ax.errorbar(x, mean, yerr=sigma, fmt="none", ecolor="black", elinewidth=0.65, capsize=2.0, zorder=3)
    return mean


def plot_convergence(metrics: list[Metrics]) -> Path:
    colors = ("#0072B2", "#56B4E9", "#009E73", "#D55E00")
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    for item, color in zip(metrics, colors):
        mean = np.mean(item.errors_deg, axis=0)
        sigma = np.std(item.errors_deg, axis=0, ddof=1)
        ax.plot(item.time_s / 60.0, mean, color=color, linewidth=1.1, label=item.label)
        ax.fill_between(item.time_s / 60.0, np.maximum(0.1, mean - sigma), mean + sigma,
                        color=color, alpha=0.10, linewidth=0.0)
    ax.axhline(POINTING_LIMIT_DEG, color="#555555", linestyle="--", linewidth=0.75, label="5° requirement")
    ax.set_yscale("log")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Attitude error [deg]")
    ax.set_title("6U capability degradation after wheel loss (mean ± 1σ)")
    ax.grid(True, which="both")
    ax.legend(loc="best", ncol=2, fontsize=5.8)
    fig.tight_layout()
    path = OUTPUT_DIR / "6u_wheel_loss_convergence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_summary(metrics: list[Metrics]) -> Path:
    labels = [item.label for item in metrics]
    x = np.arange(len(labels))
    colors = ["#0072B2", "#56B4E9", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), constrained_layout=True)

    mean_settle = _bar_with_sigma(axes[0, 0], x, [item.settle_s / 60.0 for item in metrics], colors)
    for index, item in enumerate(metrics):
        axes[0, 0].text(index, mean_settle[index] + 2.0, f"{np.count_nonzero(item.settled)}/10",
                        ha="center", va="bottom", fontsize=5.7)
    axes[0, 0].set_ylabel("Time to sustained 5° [min]")
    axes[0, 0].set_title("A. Slewing capability")

    _bar_with_sigma(axes[0, 1], x, [item.tail_compliance for item in metrics], colors, scale=100.0)
    axes[0, 1].set_ylabel("Tail samples below 5° [%]")
    axes[0, 1].set_ylim(0.0, 105.0)
    axes[0, 1].set_title("B. Disturbance rejection")

    _bar_with_sigma(axes[1, 0], x, [item.tail_rms_deg for item in metrics], colors)
    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(POINTING_LIMIT_DEG, color="#555555", linestyle="--", linewidth=0.7)
    axes[1, 0].set_ylabel("Tail RMS error [deg]")
    axes[1, 0].set_title("C. Final pointing accuracy")

    _bar_with_sigma(axes[1, 1], x, [item.orbit_uptime for item in metrics], colors, scale=100.0)
    axes[1, 1].set_ylabel("One-orbit useful pointing [%]")
    axes[1, 1].set_ylim(0.0, 105.0)
    axes[1, 1].set_title("D. Mission uptime")

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.55)
    path = OUTPUT_DIR / "6u_degradation_metric_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_momentum_headroom(metrics: list[Metrics]) -> Path:
    wheel_metrics = metrics[:3]
    labels = [item.label for item in wheel_metrics]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    _bar_with_sigma(ax, x, [item.momentum_headroom for item in wheel_metrics],
                    ["#0072B2", "#56B4E9", "#009E73"], scale=100.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Minimum wheel headroom [%]")
    ax.set_ylim(0.0, 105.0)
    ax.set_title("Momentum-storage margin over one orbit")
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = OUTPUT_DIR / "6u_momentum_headroom_by_architecture.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_operational_recovery() -> Path:
    methods, desat_s, _ = operations.calibrate()
    demand = np.linspace(0.0, 1.0, 241)
    colors = ("#D55E00", "#009E73", "#0072B2")
    uptime = []
    for method in methods:
        value, _ = operations.useful_uptime(
            method, desat_s, demand, operations.REFERENCE_HMAX_NMS, 635.0
        )
        uptime.append(value)
    uptime = np.asarray(uptime)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for method, color, values in zip(methods, colors, uptime):
        axes[0].plot(demand, values, color=color, linewidth=1.2, label=method.name)
    axes[0].axhline(90.0, color="#555555", linestyle="--", linewidth=0.7)
    axes[0].set_xlabel("Slew demand [h$^{-1}$]")
    axes[0].set_ylabel("Useful uptime [%]")
    axes[0].set_ylim(0.0, 102.0)
    axes[0].set_title("A. Repeated operations")
    axes[0].legend(fontsize=5.2, loc="upper right")

    excursions = 1.0e3 * np.asarray([method.peak_delta_h_nms for method in methods])
    bars = axes[1].bar(np.arange(3), excursions, color=colors, edgecolor="black", linewidth=0.35)
    axes[1].axhline(7.5, color="#555555", linestyle="--", linewidth=0.7,
                    label="25–75% band")
    axes[1].set_xticks(np.arange(3), ["Reactive", r"$\gamma=0.5$", "Planner"], rotation=16)
    axes[1].set_ylabel(r"Peak $|\Delta h|$ per slew [mN m s]")
    axes[1].set_title("B. Momentum deposited")
    axes[1].legend(fontsize=5.2)
    for bar, value in zip(bars, excursions):
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 0.16, f"{value:.2f}",
                     ha="center", fontsize=5.4)

    denominator = np.maximum(100.0 - uptime[0], 1e-9)
    for index, (label, color) in enumerate(((r"$\gamma=0.5$", colors[1]), ("Planner", colors[2])), start=1):
        recovered = 100.0 * np.clip((uptime[index] - uptime[0]) / denominator, 0.0, 1.0)
        axes[2].plot(demand, recovered, color=color, linewidth=1.2, label=label)
    axes[2].set_xlabel("Slew demand [h$^{-1}$]")
    axes[2].set_ylabel("Lost uptime recovered [%]")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].set_title("C. 3+1 capability recovery")
    axes[2].legend(fontsize=5.5, loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.55)
    fig.suptitle("Recovering degraded 3+1 mission capability at 635 km")
    path = OUTPUT_DIR / "6u_3p1_operational_recovery.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metrics(metrics: list[Metrics]) -> Path:
    path = OUTPUT_DIR / "6u_degradation_metrics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "architecture", "settled_runs", "mean_settle_min", "tail_compliance_pct",
            "tail_rms_deg", "one_orbit_uptime_pct", "momentum_headroom_pct",
        ])
        for item in metrics:
            writer.writerow([
                item.label, int(np.count_nonzero(item.settled)), f"{np.mean(item.settle_s)/60.0:.4f}",
                f"{100*np.mean(item.tail_compliance):.4f}", f"{np.mean(item.tail_rms_deg):.6f}",
                f"{100*np.mean(item.orbit_uptime):.4f}",
                "" if np.all(np.isnan(item.momentum_headroom)) else f"{100*np.nanmean(item.momentum_headroom):.4f}",
            ])
    return path


def main() -> None:
    configure_ieee_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = architectures()
    metrics = [measure(spec) for spec in specs]
    outputs = [
        plot_convergence(metrics),
        plot_metric_summary(metrics),
        plot_momentum_headroom(metrics),
        plot_operational_recovery(),
        save_metrics(metrics),
    ]
    print("Architecture metrics:")
    for item in metrics:
        print(
            f"  {item.label}: settled={np.count_nonzero(item.settled)}/10, "
            f"mean settle={np.mean(item.settle_s)/60.0:.2f} min, "
            f"tail compliance={100*np.mean(item.tail_compliance):.1f}%, "
            f"tail RMS={np.mean(item.tail_rms_deg):.3f} deg, "
            f"orbit uptime={100*np.mean(item.orbit_uptime):.1f}%"
        )
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
