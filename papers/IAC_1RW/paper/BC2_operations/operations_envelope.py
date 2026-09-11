"""Reduced-order 3+1 operational-envelope model calibrated from saved runs.

No new Monte Carlo simulations are performed.  The script extracts random-slew
and desaturation statistics from existing archives, then evaluates repeated
mission operations analytically over slew demand, wheel sizing, altitude, and
actuator authority.
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


REACTIVE_ARCHIVE = PAPER_DIR / "6U_mode_switching_disturbance/outputs/6u_mode_switching_disturbance_mc_10_1000min_20260911_041808.sim"
COALLOCATION_ARCHIVE = PAPER_DIR / "6U_coallocation/outputs/6u_coallocation_disturbance_gamma_0p50_mc_10_180min_20260911_045348.sim"
PLANNER_ARCHIVE = PAPER_DIR / "6U_planner/outputs/6u_3mtq_1rw_boresight_planner_mc_10_20260911_131519.sim"

POINTING_LIMIT_DEG = 5.0
SETTLE_DWELL_S = 120.0
MISSION_HORIZON_H = 24.0
REFERENCE_ALTITUDE_KM = 400.0
REFERENCE_HMAX_NMS = 15.0e-3
REFERENCE_RW_TORQUE_NM = 2.0e-3
REFERENCE_MTQ_DIPOLE_AM2 = 0.6
DESAT_ENTRY = 0.75
DESAT_EXIT = 0.25
BUS_SCALE_6U = 2.0 ** (1.0 / 3.0)

# Same first-principles environment model used in Case I.
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
    pointing_quality: float
    successful_slews: int
    total_slews: int


def _full_attitude_error_deg(run) -> np.ndarray:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    target = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.einsum("ij,ij->i", q, target)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _planner_boresight_error_deg(run, config: dict) -> np.ndarray:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    goal = np.asarray(config["config"]["goal_vec"], dtype=float)
    body_z = np.array([0.0, 0.0, 1.0])
    actual = np.asarray([rot_mat(quaternion) @ body_z for quaternion in q])
    return np.rad2deg(np.arccos(np.clip(actual @ goal, -1.0, 1.0)))


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


def _calibrate_slews(path: Path, name: str, *, planner: bool = False) -> Calibration:
    results = ADCS.SimulationResults.load(path)
    settle_times = []
    peak_excursions = []
    quality = []
    successes = 0
    for index, run in enumerate(results.runs):
        time_s = np.asarray(run.time_s, dtype=float)
        error = (
            _planner_boresight_error_deg(run, results.configs[index])
            if planner else _full_attitude_error_deg(run)
        )
        settle_index, success = _settling_index(time_s, error)
        successes += int(success)
        settle_times.append(float(time_s[settle_index]))
        momentum = np.asarray([state.h[0] for state in run.state_hist], dtype=float)
        peak_excursions.append(float(np.max(np.abs(momentum[: settle_index + 1] - momentum[0]))))
        tail = max(1, int(np.ceil(0.20 * len(error))))
        quality.append(float(np.mean(error[-tail:] <= POINTING_LIMIT_DEG)))
    return Calibration(
        name=name,
        settle_s=float(np.mean(settle_times)),
        peak_delta_h_nms=float(np.mean(peak_excursions)),
        pointing_quality=float(np.mean(quality)),
        successful_slews=successes,
        total_slews=len(results.runs),
    )


def _mode_mask(momentum_fraction: np.ndarray) -> np.ndarray:
    desaturating = False
    output = np.zeros(len(momentum_fraction), dtype=bool)
    for index, fraction in enumerate(momentum_fraction):
        if not desaturating and fraction >= DESAT_ENTRY:
            desaturating = True
        elif desaturating and fraction <= DESAT_EXIT:
            desaturating = False
        output[index] = desaturating
    return output


def _calibrate_desaturation_s(path: Path) -> tuple[float, int]:
    results = ADCS.SimulationResults.load(path)
    durations = []
    for run in results.runs:
        time_s = np.asarray(run.time_s, dtype=float)
        momentum_fraction = np.abs(
            np.asarray([state.h[0] for state in run.state_hist], dtype=float)
        ) / REFERENCE_HMAX_NMS
        mode = _mode_mask(momentum_fraction)
        changes = np.diff(np.r_[False, mode, False].astype(int))
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
    methods = [
        _calibrate_slews(REACTIVE_ARCHIVE, "Reactive 75/25"),
        _calibrate_slews(COALLOCATION_ARCHIVE, r"Co-allocation $\gamma=0.5$"),
        _calibrate_slews(PLANNER_ARCHIVE, "Feasibility-aware planner", planner=True),
    ]
    desat_s, episodes = _calibrate_desaturation_s(REACTIVE_ARCHIVE)
    return methods, desat_s, episodes


def magnetic_field_t(altitude_km: np.ndarray) -> np.ndarray:
    return B_400_T * ((R_E_KM + REFERENCE_ALTITUDE_KM) / (R_E_KM + altitude_km)) ** 3


def disturbance_torque_nm(altitude_km: np.ndarray) -> np.ndarray:
    """Case-I disturbance model evaluated at an equivalent-volume 6U scale."""
    scale = BUS_SCALE_6U
    radius_m = (R_E_KM + altitude_km) * 1.0e3
    velocity_m_s = np.sqrt(MU_M3_S2 / radius_m)
    density = RHO_400 * np.exp(-(altitude_km - 400.0) / RHO_SCALE_HEIGHT_KM)
    gg = 3.0 * MU_M3_S2 / radius_m**3 * (0.02 * scale**5)
    drag = 0.5 * density * velocity_m_s**2 * 2.2 * (0.03 * scale**2) * (0.05 * scale)
    srp = 4.56e-6 * 1.3 * (0.03 * scale**2) * (0.05 * scale)
    residual_dipole = (0.01 * scale**3) * magnetic_field_t(altitude_km)
    return gg + drag + srp + residual_dipole


def altitude_load_factor(altitude_km: np.ndarray) -> np.ndarray:
    """Normalize Case-I disturbance-to-magnetic-authority demand to 400 km."""
    demand = disturbance_torque_nm(altitude_km) / magnetic_field_t(altitude_km)
    reference = disturbance_torque_nm(np.asarray(REFERENCE_ALTITUDE_KM)) / B_400_T
    return demand / reference


def useful_uptime(
    calibration: Calibration,
    desat_reference_s: float,
    slew_demand_per_h: np.ndarray,
    hmax_nms: np.ndarray,
    altitude_km: np.ndarray,
    *,
    rw_torque_scale: np.ndarray | float = 1.0,
    mtq_dipole_scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return useful-pointing uptime and expected emergency dumps per day.

    Repeated random slews are conservatively chained using the empirically
    observed mean peak wheel excursion.  Co-allocation and planning therefore
    benefit directly from their smaller measured excursion.  A dump is charged
    only after the first 25%->75% storage band is consumed.
    """
    mission_s = MISSION_HORIZON_H * 3600.0
    slew_count = np.asarray(slew_demand_per_h) * MISSION_HORIZON_H
    wheel_capacity = np.asarray(hmax_nms)
    load_per_slew = calibration.peak_delta_h_nms * altitude_load_factor(np.asarray(altitude_km))
    total_load = slew_count * load_per_slew
    threshold_band = (DESAT_ENTRY - DESAT_EXIT) * wheel_capacity
    expected_dumps = np.maximum(0.0, total_load - threshold_band) / np.maximum(threshold_band, 1e-12)

    slew_time = calibration.settle_s / np.maximum(np.asarray(rw_torque_scale), 1e-6)
    desat_time = (
        desat_reference_s
        * (wheel_capacity / REFERENCE_HMAX_NMS)
        * (B_400_T / magnetic_field_t(np.asarray(altitude_km)))
        / np.maximum(np.asarray(mtq_dipole_scale), 1e-6)
    )
    downtime_s = slew_count * slew_time + expected_dumps * desat_time
    uptime = calibration.pointing_quality * np.clip(1.0 - downtime_s / mission_s, 0.0, 1.0)
    return 100.0 * uptime, expected_dumps


def _contoured_panel(ax, x, y, values, *, xlabel: str, ylabel: str, title: str):
    image = ax.pcolormesh(x, y, values, shading="auto", cmap="RdYlGn", vmin=0.0, vmax=100.0)
    contours = ax.contour(x, y, values, levels=[50.0, 75.0, 90.0], colors="black", linewidths=0.55)
    ax.clabel(contours, fmt="%g%%", fontsize=5.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return image


def plot_wheel_envelope(methods: list[Calibration], desat_s: float) -> Path:
    demand = np.linspace(0.0, 1.0, 181)
    hmax = np.linspace(3.0e-3, 30.0e-3, 181)
    demand_grid, hmax_grid = np.meshgrid(demand, hmax)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, method in zip(axes, methods):
        uptime, _ = useful_uptime(method, desat_s, demand_grid, hmax_grid, REFERENCE_ALTITUDE_KM)
        image = _contoured_panel(
            ax, demand, 1.0e3 * hmax, uptime,
            xlabel="Slew demand [h$^{-1}$]", ylabel=r"Wheel capacity $h_{max}$ [mN m s]",
            title=method.name,
        )
        ax.axhline(1.0e3 * REFERENCE_HMAX_NMS, color="#0072B2", linestyle="--", linewidth=0.7)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Useful pointing uptime [%]")
    fig.suptitle("Repeated-slew operational envelope at 400 km")
    path = OUTPUT_DIR / "uptime_vs_slew_demand_and_wheel_capacity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_altitude_envelope(methods: list[Calibration], desat_s: float) -> Path:
    demand = np.linspace(0.0, 1.0, 181)
    altitude = np.linspace(300.0, 1200.0, 181)
    demand_grid, altitude_grid = np.meshgrid(demand, altitude)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, method in zip(axes, methods):
        uptime, _ = useful_uptime(
            method, desat_s, demand_grid, REFERENCE_HMAX_NMS, altitude_grid
        )
        image = _contoured_panel(
            ax, demand, altitude, uptime,
            xlabel="Slew demand [h$^{-1}$]", ylabel="Altitude [km]", title=method.name,
        )
        ax.axhline(REFERENCE_ALTITUDE_KM, color="#0072B2", linestyle="--", linewidth=0.7)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Useful pointing uptime [%]")
    fig.suptitle("Altitude sensitivity ($h_{max}=15$ mN m s, $m_{max}=0.6$ A m$^2$)")
    path = OUTPUT_DIR / "uptime_vs_slew_demand_and_altitude.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_authority_envelope(methods: list[Calibration], desat_s: float) -> Path:
    rw_scale = np.linspace(0.5, 2.0, 161)
    mtq_scale = np.linspace(0.5, 2.0, 161)
    rw_grid, mtq_grid = np.meshgrid(rw_scale, mtq_scale)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, method in zip(axes, methods):
        uptime, _ = useful_uptime(
            method, desat_s, 0.5, REFERENCE_HMAX_NMS, REFERENCE_ALTITUDE_KM,
            rw_torque_scale=rw_grid, mtq_dipole_scale=mtq_grid,
        )
        image = _contoured_panel(
            ax, rw_scale, mtq_scale, uptime,
            xlabel="RW torque / 2 mN m", ylabel="MTQ dipole / 0.6 A m$^2$", title=method.name,
        )
        ax.scatter(1.0, 1.0, marker="*", s=34, color="#0072B2", edgecolor="black", linewidth=0.3)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("Useful pointing uptime [%]")
    fig.suptitle("Actuator-authority sensitivity (0.5 slews/h, 400 km, 15 mN m s)")
    path = OUTPUT_DIR / "uptime_vs_rw_and_mtq_authority.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_representative_comparison(methods: list[Calibration], desat_s: float) -> Path:
    """Summarize the mechanism at the BC2/CanX-2-like 635 km operating point."""
    demand = np.linspace(0.0, 1.0, 241)
    colors = ("#D55E00", "#009E73", "#0072B2")
    fig, (ax_uptime, ax_dumps) = plt.subplots(
        2, 1, figsize=(3.55, 4.15), sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0]},
    )
    for method, color in zip(methods, colors):
        uptime, dumps = useful_uptime(
            method, desat_s, demand, REFERENCE_HMAX_NMS, 635.0
        )
        ax_uptime.plot(demand, uptime, color=color, linewidth=1.35, label=method.name)
        ax_dumps.plot(demand, dumps, color=color, linewidth=1.25, label=method.name)

    ax_uptime.axhline(90.0, color="#555555", linestyle="--", linewidth=0.7,
                      label="90% useful uptime")
    ax_uptime.set_ylabel("Useful pointing uptime [%]")
    ax_uptime.set_ylim(0.0, 102.0)
    ax_uptime.set_title("Repeated-slew operations at 635 km")
    ax_uptime.grid(True)
    ax_uptime.legend(loc="upper right", fontsize=5.5)

    ax_dumps.set_xlabel("Slew demand [h$^{-1}$]")
    ax_dumps.set_ylabel("Expected emergency\ndumps per day")
    ax_dumps.set_ylim(bottom=0.0)
    ax_dumps.grid(True)
    fig.tight_layout()
    path = OUTPUT_DIR / "representative_uptime_and_desaturation_burden.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary(methods: list[Calibration], desat_s: float, episodes: int) -> Path:
    path = OUTPUT_DIR / "operations_calibration.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "mean_settle_s", "mean_peak_delta_h_mNms", "tail_pointing_quality",
            "successful_slews", "total_slews", "desat_75_to_25_mean_s", "desat_episodes",
        ])
        for method in methods:
            writer.writerow([
                method.name.replace("$", ""), f"{method.settle_s:.3f}",
                f"{1e3 * method.peak_delta_h_nms:.6f}", f"{method.pointing_quality:.6f}",
                method.successful_slews, method.total_slews, f"{desat_s:.3f}", episodes,
            ])
    return path


def main() -> None:
    configure_ieee_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods, desat_s, episodes = calibrate()
    outputs = [
        plot_wheel_envelope(methods, desat_s),
        plot_altitude_envelope(methods, desat_s),
        plot_authority_envelope(methods, desat_s),
        plot_representative_comparison(methods, desat_s),
        save_summary(methods, desat_s, episodes),
    ]
    print("Calibration from saved simulation archives:")
    for method in methods:
        print(
            f"  {method.name}: settle={method.settle_s/60.0:.2f} min, "
            f"peak |Delta h|={1e3*method.peak_delta_h_nms:.3f} mN m s, "
            f"tail quality={100*method.pointing_quality:.1f}%, "
            f"settled={method.successful_slews}/{method.total_slews}"
        )
    print(f"  Desaturation 75%->25%: {desat_s/60.0:.2f} min ({episodes} complete episodes)")
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
