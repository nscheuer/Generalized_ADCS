"""Analytical 6U repeated-slew envelope calibrated from saved simulations.

This companion to ``operations_envelope.py`` retains the analytical sweep over
slew demand, wheel capacity, altitude, and actuator authority.  Its output is
labelled as estimated time in <=5 degree pointing: the measured fraction of a
saved run within 5 degrees, reduced by modeled slew and desaturation downtime.
"""

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
OUTPUT_DIR = SCRIPT_DIR / "outputs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from plot_style import configure_ieee_style


SIX_U_MODE_DIR = PAPER_DIR / "6U_mode_switching_disturbance/outputs"
SIX_U_COALLOCATION_DIR = PAPER_DIR / "6U_coallocation/outputs"
SIX_U_PLANNER_DIR = PAPER_DIR / "6U_planner/outputs"
SIX_U_MODE_PREFIX = "6u_mode_switching_disturbance_mc_10_"
SIX_U_COALLOCATION_PREFIX = "6u_coallocation_disturbance_gamma_0p50_"
SIX_U_PLANNER_PREFIX = "6u_3mtq_1rw_boresight_planner_mc_10"

POINTING_LIMIT_DEG = 5.0
SETTLE_DWELL_S = 120.0
MISSION_HORIZON_H = 24.0
AVERAGE_SLEW_DEG = 109.5
REFERENCE_ALTITUDE_KM = 400.0
REFERENCE_HMAX_NMS = 15.0e-3
REFERENCE_RW_TORQUE_NM = 2.0e-3
REFERENCE_MTQ_DIPOLE_AM2 = 0.6
DESAT_ENTRY = 0.75
DESAT_EXIT = 0.25
SPACECRAFT_SCALE_6U = 2.0 ** (1.0 / 3.0)

R_E_KM = 6378.137
MU_M3_S2 = 398600.4418e9
B_400_T = 35.0e-6
RHO_400 = 3.9e-12
RHO_SCALE_HEIGHT_KM = 60.0


@dataclass(frozen=True)
class Calibration:
    name: str
    settle_s: float
    peak_delta_h_nms: float
    time_in_5deg_fraction: float
    successful_slews: int
    total_slews: int


def _latest_archive(directory: Path, prefix: str) -> Path:
    matches = list(directory.glob(f"{prefix}*.sim"))
    if not matches:
        raise FileNotFoundError(
            f"No 6U simulation archive matching {prefix!r} in {directory}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _full_attitude_error_deg(run) -> np.ndarray:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    target = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.einsum("ij,ij->i", q, target), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(np.abs(dots)))


def _planner_boresight_error_deg(run, config: dict) -> np.ndarray:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    goal /= np.linalg.norm(goal)
    body_z = np.array([0.0, 0.0, 1.0])
    boresight = np.asarray([rot_mat(quaternion) @ body_z for quaternion in q])
    return np.rad2deg(np.arccos(np.clip(boresight @ goal, -1.0, 1.0)))


def _settling_index(time_s: np.ndarray, error_deg: np.ndarray) -> tuple[int, bool]:
    dt_s = float(np.median(np.diff(time_s)))
    dwell_samples = max(1, int(np.ceil(SETTLE_DWELL_S / dt_s)))
    rolling = np.convolve(
        (error_deg <= POINTING_LIMIT_DEG).astype(int),
        np.ones(dwell_samples, dtype=int),
        mode="valid",
    )
    candidates = np.flatnonzero(rolling == dwell_samples)
    if candidates.size:
        return int(candidates[0]), True
    return len(error_deg) - 1, False


def _time_in_bound_fraction(time_s: np.ndarray, error_deg: np.ndarray) -> float:
    duration_s = time_s[-1] - time_s[0]
    return float(np.sum(np.diff(time_s) * (error_deg[:-1] <= POINTING_LIMIT_DEG)) / duration_s)


def _calibrate_slews(path: Path, name: str, *, planner: bool = False) -> Calibration:
    results = ADCS.SimulationResults.load(path)
    settle_times = []
    peak_excursions = []
    fractions = []
    successes = 0
    for index, run in enumerate(results.runs):
        time_s = np.asarray(run.time_s, dtype=float)
        error_deg = (
            _planner_boresight_error_deg(run, results.configs[index])
            if planner else _full_attitude_error_deg(run)
        )
        settle_index, success = _settling_index(time_s, error_deg)
        successes += int(success)
        settle_times.append(float(time_s[settle_index]))
        momentum = np.asarray([state.h[0] for state in run.state_hist], dtype=float)
        peak_excursions.append(float(np.max(np.abs(momentum[:settle_index + 1] - momentum[0]))))
        fractions.append(_time_in_bound_fraction(time_s, error_deg))
    return Calibration(
        name=name,
        settle_s=float(np.mean(settle_times)),
        peak_delta_h_nms=float(np.mean(peak_excursions)),
        time_in_5deg_fraction=float(np.mean(fractions)),
        successful_slews=successes,
        total_slews=len(results.runs),
    )


def _mode_mask(momentum_fraction: np.ndarray) -> np.ndarray:
    desaturating = False
    mode = np.zeros(len(momentum_fraction), dtype=bool)
    for index, fraction in enumerate(momentum_fraction):
        if not desaturating and fraction >= DESAT_ENTRY:
            desaturating = True
        elif desaturating and fraction <= DESAT_EXIT:
            desaturating = False
        mode[index] = desaturating
    return mode


def _calibrate_desaturation_s(path: Path) -> tuple[float, int]:
    results = ADCS.SimulationResults.load(path)
    durations = []
    for run in results.runs:
        time_s = np.asarray(run.time_s, dtype=float)
        fraction = np.abs(np.asarray([state.h[0] for state in run.state_hist])) / REFERENCE_HMAX_NMS
        changes = np.diff(np.r_[False, _mode_mask(fraction), False].astype(int))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        durations.extend(
            float(time_s[end] - time_s[start])
            for start, end in zip(starts, ends)
            if end > start
        )
    if not durations:
        raise RuntimeError("No complete 75% to 25% desaturation episodes found.")
    return float(np.mean(durations)), len(durations)


def calibrate() -> tuple[list[Calibration], float, int]:
    reactive_archive = _latest_archive(SIX_U_MODE_DIR, SIX_U_MODE_PREFIX)
    coallocation_archive = _latest_archive(SIX_U_COALLOCATION_DIR, SIX_U_COALLOCATION_PREFIX)
    planner_archive = _latest_archive(SIX_U_PLANNER_DIR, SIX_U_PLANNER_PREFIX)
    methods = [
        _calibrate_slews(reactive_archive, "6U reactive 75/25"),
        _calibrate_slews(coallocation_archive, r"6U co-allocation $\gamma=0.5$"),
        _calibrate_slews(planner_archive, "6U feasibility-aware planner", planner=True),
    ]
    desat_s, episodes = _calibrate_desaturation_s(reactive_archive)
    return methods, desat_s, episodes


def magnetic_field_t(altitude_km: np.ndarray) -> np.ndarray:
    return B_400_T * ((R_E_KM + REFERENCE_ALTITUDE_KM) / (R_E_KM + altitude_km)) ** 3


def disturbance_torque_nm(altitude_km: np.ndarray) -> np.ndarray:
    """Disturbance model for the reference 6U spacecraft."""
    scale = SPACECRAFT_SCALE_6U
    radius_m = (R_E_KM + altitude_km) * 1.0e3
    velocity_m_s = np.sqrt(MU_M3_S2 / radius_m)
    density = RHO_400 * np.exp(-(altitude_km - 400.0) / RHO_SCALE_HEIGHT_KM)
    gg = 3.0 * MU_M3_S2 / radius_m**3 * (0.02 * scale**5)
    drag = 0.5 * density * velocity_m_s**2 * 2.2 * (0.03 * scale**2) * (0.05 * scale)
    srp = 4.56e-6 * 1.3 * (0.03 * scale**2) * (0.05 * scale)
    residual_dipole = (0.01 * scale**3) * magnetic_field_t(altitude_km)
    return gg + drag + srp + residual_dipole


def altitude_load_factor(altitude_km: np.ndarray) -> np.ndarray:
    demand = disturbance_torque_nm(altitude_km) / magnetic_field_t(altitude_km)
    reference = disturbance_torque_nm(np.asarray(REFERENCE_ALTITUDE_KM)) / B_400_T
    return demand / reference


def estimated_time_in_5deg_pointing(
    calibration: Calibration,
    desat_reference_s: float,
    slew_demand_deg_per_h: np.ndarray | float,
    hmax_nms: np.ndarray | float,
    altitude_km: np.ndarray | float,
    *,
    rw_torque_scale: np.ndarray | float = 1.0,
    mtq_dipole_scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 24-hour time within 5 deg and desaturation episodes.

    ``slew_demand_deg_per_h`` is converted to average slew count using the
    measured 109.5 deg mean rotation per slew. The direct simulation fraction
    within 5 deg is then reduced only by modeled acquisition and desaturation
    time over the 24-hour horizon.
    """
    mission_s = MISSION_HORIZON_H * 3600.0
    slew_count = np.asarray(slew_demand_deg_per_h) * MISSION_HORIZON_H / AVERAGE_SLEW_DEG
    capacity = np.asarray(hmax_nms)
    load_per_slew = calibration.peak_delta_h_nms * altitude_load_factor(np.asarray(altitude_km))
    total_load = slew_count * load_per_slew
    storage_band = (DESAT_ENTRY - DESAT_EXIT) * capacity
    desaturation_episodes = np.maximum(0.0, total_load - storage_band) / np.maximum(storage_band, 1e-12)
    slew_time = calibration.settle_s / np.maximum(np.asarray(rw_torque_scale), 1e-6)
    desaturation_time = (
        desat_reference_s
        * (capacity / REFERENCE_HMAX_NMS)
        * (B_400_T / magnetic_field_t(np.asarray(altitude_km)))
        / np.maximum(np.asarray(mtq_dipole_scale), 1e-6)
    )
    unavailable_fraction = np.clip(
        (slew_count * slew_time + desaturation_episodes * desaturation_time) / mission_s,
        0.0,
        1.0,
    )
    return 100.0 * calibration.time_in_5deg_fraction * (1.0 - unavailable_fraction), desaturation_episodes


def _panel(ax, x, y, value, *, xlabel: str, ylabel: str, title: str):
    image = ax.pcolormesh(x, y, value, shading="auto", cmap="RdYlGn", vmin=0.0, vmax=100.0)
    contours = ax.contour(x, y, value, levels=[50.0, 75.0, 90.0], colors="black", linewidths=0.55)
    ax.clabel(contours, fmt="%g%%", fontsize=5.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return image


def plot_wheel_envelope(methods: list[Calibration], desat_s: float) -> Path:
    demand = np.linspace(0.0, AVERAGE_SLEW_DEG, 181)
    hmax = np.linspace(3.0e-3, 30.0e-3, 181)
    demand_grid, hmax_grid = np.meshgrid(demand, hmax)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        time_5deg, _ = estimated_time_in_5deg_pointing(method, desat_s, demand_grid, hmax_grid, REFERENCE_ALTITUDE_KM)
        image = _panel(ax, demand, 1e3 * hmax, time_5deg,
                       xlabel="Slew demand [deg h$^{-1}$]", ylabel=r"Wheel capacity $h_{max}$ [mN m s]", title=method.name)
        ax.axhline(1e3 * REFERENCE_HMAX_NMS, color="#0072B2", linestyle="--", linewidth=0.7)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Estimated time in ≤5° pointing [% of day]")
    fig.suptitle("6U repeated-slew envelope at 400 km")
    path = OUTPUT_DIR / "uptime_vs_slew_demand_and_wheel_capacity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_altitude_envelope(methods: list[Calibration], desat_s: float) -> Path:
    demand = np.linspace(0.0, AVERAGE_SLEW_DEG, 181)
    altitude = np.linspace(300.0, 1200.0, 181)
    demand_grid, altitude_grid = np.meshgrid(demand, altitude)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        time_5deg, _ = estimated_time_in_5deg_pointing(method, desat_s, demand_grid, REFERENCE_HMAX_NMS, altitude_grid)
        image = _panel(ax, demand, altitude, time_5deg,
                       xlabel="Slew demand [deg h$^{-1}$]", ylabel="Altitude [km]", title=method.name)
        ax.axhline(REFERENCE_ALTITUDE_KM, color="#0072B2", linestyle="--", linewidth=0.7)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Estimated time in ≤5° pointing [% of day]")
    fig.suptitle("6U altitude sensitivity ($h_{max}=15$ mN m s, $m_{max}=0.6$ A m$^2$)")
    path = OUTPUT_DIR / "uptime_vs_slew_demand_and_altitude.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_authority_envelope(methods: list[Calibration], desat_s: float) -> Path:
    rw_scale = np.linspace(0.5, 2.0, 161)
    mtq_scale = np.linspace(0.5, 2.0, 161)
    rw_grid, mtq_grid = np.meshgrid(rw_scale, mtq_scale)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        time_5deg, _ = estimated_time_in_5deg_pointing(
            method, desat_s, 0.5 * AVERAGE_SLEW_DEG, REFERENCE_HMAX_NMS, REFERENCE_ALTITUDE_KM,
            rw_torque_scale=rw_grid, mtq_dipole_scale=mtq_grid,
        )
        image = _panel(ax, rw_scale, mtq_scale, time_5deg,
                       xlabel="RW torque / 2 mN m", ylabel="MTQ dipole / 0.6 A m$^2$", title=method.name)
        ax.scatter(1.0, 1.0, marker="*", s=34, color="#0072B2", edgecolor="black", linewidth=0.3)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Estimated time in ≤5° pointing [% of day]")
    fig.suptitle("6U actuator-authority sensitivity (54.75 deg/h, 400 km, 15 mN m s)")
    path = OUTPUT_DIR / "uptime_vs_rw_and_mtq_authority.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_momentum_deposited(methods: list[Calibration]) -> Path:
    """Plot the observed peak wheel-momentum excursion for one slew."""
    values = 1.0e3 * np.asarray([method.peak_delta_h_nms for method in methods])
    labels = [method.name.replace("6U ", "") for method in methods]
    colors = ("#D55E00", "#009E73", "#0072B2")
    positions = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    bars = ax.bar(
        positions, values, color=colors, edgecolor="black", linewidth=0.55
    )
    storage_band_mnms = 1.0e3 * (DESAT_ENTRY - DESAT_EXIT) * REFERENCE_HMAX_NMS
    ax.axhline(
        storage_band_mnms, color="#555555", linestyle="--", linewidth=0.75,
        label="25–75% wheel-storage band",
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.14, f"{value:.2f}",
            ha="center", va="bottom", fontsize=6.5,
        )

    ax.set_xticks(positions, labels)
    ax.set_ylabel(r"Peak $|\Delta h|$ per representative slew [mN m s]")
    ax.set_ylim(0.0, max(storage_band_mnms * 1.16, np.max(values) * 1.28))
    ax.set_title("6U wheel momentum deposited by one slew")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=6.0)
    fig.tight_layout()
    path = OUTPUT_DIR / "momentum_deposited_per_slew.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_representative_comparison(methods: list[Calibration], desat_s: float) -> Path:
    demand = np.linspace(0.0, AVERAGE_SLEW_DEG, 241)
    colors = ("#D55E00", "#009E73", "#0072B2")
    fig, (ax_time, ax_desaturation) = plt.subplots(2, 1, figsize=(3.55, 4.15), sharex=True,
                                                     gridspec_kw={"height_ratios": [1.65, 1.0]})
    for method, color in zip(methods, colors):
        time_5deg, episodes = estimated_time_in_5deg_pointing(method, desat_s, demand, REFERENCE_HMAX_NMS, 635.0)
        ax_time.plot(demand, time_5deg, color=color, linewidth=1.35, label=method.name)
        ax_desaturation.plot(demand, episodes, color=color, linewidth=1.25, label=method.name)
    ax_time.set_ylabel("Estimated time in ≤5°\npointing [% of day]")
    ax_time.set_ylim(0.0, 102.0)
    ax_time.set_title("6U repeated-slew operations at 635 km")
    ax_time.grid(True)
    ax_time.legend(loc="upper right", fontsize=5.5)
    ax_desaturation.set_xlabel("Slew demand [deg h$^{-1}$]")
    ax_desaturation.set_ylabel("Expected desaturation\nmode episodes per day")
    ax_desaturation.set_ylim(bottom=0.0)
    ax_desaturation.grid(True)
    fig.tight_layout()
    path = OUTPUT_DIR / "representative_uptime_and_desaturation_burden.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary(methods: list[Calibration], desat_s: float, episodes: int) -> Path:
    path = OUTPUT_DIR / "analytical_operations_calibration.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "mean_settle_s", "mean_peak_delta_h_mNms", "time_in_5deg_percent", "successful_slews", "total_slews", "desat_75_to_25_mean_s", "desat_episodes"])
        for method in methods:
            writer.writerow([method.name.replace("$", ""), f"{method.settle_s:.3f}", f"{1e3 * method.peak_delta_h_nms:.6f}", f"{100 * method.time_in_5deg_fraction:.6f}", method.successful_slews, method.total_slews, f"{desat_s:.3f}", episodes])
    return path


def main() -> None:
    configure_ieee_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods, desat_s, episodes = calibrate()
    outputs = [
        plot_wheel_envelope(methods, desat_s),
        plot_altitude_envelope(methods, desat_s),
        plot_authority_envelope(methods, desat_s),
        plot_momentum_deposited(methods),
        plot_representative_comparison(methods, desat_s),
        save_summary(methods, desat_s, episodes),
    ]
    print("Analytical 6U envelope calibrated from saved runs:")
    for method in methods:
        print(f"  {method.name}: time in ≤5°={100 * method.time_in_5deg_fraction:.1f}%, settled={method.successful_slews}/{method.total_slews}")
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
