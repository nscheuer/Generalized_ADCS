#!/usr/bin/env python3
"""Compare 3-MTQ Lovera alignment to quaternion and vector goals."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ADCS  # noqa: E402
from ADCS.helpers.math_helpers import quat_diff, rot_mat  # noqa: E402
from ADCS.orbits.universal_constants import EarthConstants  # noqa: E402


SIM_DURATION_S = 12_000.0
SIM_DT_S = 50.0
ORBIT_RADIUS_KM = 7_000.0
ERROR_THRESHOLD_DEG = 10.0

VECTOR_COLOR = "#0072B2"
QUATERNION_COLOR = "#D55E00"
GRID_COLOR = "#C7CDD3"

# A +90 degree rotation about body Y maps the +Z boresight to inertial +X.
Q_REFERENCE = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
VECTOR_REFERENCE = np.array([1.0, 0.0, 0.0])
INITIAL_STATE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def make_satellite() -> ADCS.Satellite:
    """Create the three-axis MTQ-only satellite used in Lovera tests."""
    mtqs = [ADCS.MTQ(axis=axis, max_torque=1.0) for axis in np.eye(3)]
    mtms = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
    return ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=mtqs,
        sensors=mtms,
        boresight=np.array([0.0, 0.0, 1.0]),
    )


def make_initial_orbit() -> ADCS.Orbital_State:
    """Create the inclined 7000 km orbit used by Lovera controller tests."""
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=ORBIT_RADIUS_KM * np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
    )


def run_case(goal) -> ADCS.SimulationResults:
    """Run one deterministic MTQ-only Lovera alignment simulation."""
    np.random.seed(37)
    satellite = make_satellite()
    controller = ADCS.controller.MTQ_Lovera(
        est_sat=satellite,
        p_gain=2.0e-5,
        d_gain=2.0e-2,
        eps=1.0,
    )
    return ADCS.simulate(
        x=INITIAL_STATE.copy(),
        satellite=satellite,
        controller=controller,
        goal=goal,
        os0=make_initial_orbit(),
        dt=SIM_DT_S,
        tf=SIM_DURATION_S,
    )


def quaternion_error_deg(results: ADCS.SimulationResults) -> np.ndarray:
    """Return the minimal full-attitude rotation error."""
    quaternions = np.asarray(results.first().state_hist, dtype=float)[:, 3:7]
    errors = []
    for quaternion in quaternions:
        error_quaternion = quat_diff(quaternion, Q_REFERENCE)
        scalar = np.clip(abs(error_quaternion[0]), -1.0, 1.0)
        errors.append(np.degrees(2.0 * np.arccos(scalar)))
    return np.asarray(errors)


def vector_error_deg(results: ADCS.SimulationResults) -> np.ndarray:
    """Return the body-boresight to inertial-vector angular error."""
    run = results.first()
    quaternions = np.asarray(run.state_hist, dtype=float)[:, 3:7]
    boresights = np.asarray(run.boresight_hist, dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)[:, 1:4]

    boresight_eci = np.asarray(
        [rot_mat(q) @ boresight for q, boresight in zip(quaternions, boresights)]
    )
    boresight_eci /= np.linalg.norm(boresight_eci, axis=1, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    dots = np.einsum("ni,ni->n", boresight_eci, targets)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def settling_time(time_s: np.ndarray, error_deg: np.ndarray) -> float | None:
    """Return the first time after which error remains below 10 degrees."""
    above_threshold_later = np.maximum.accumulate(
        (error_deg >= ERROR_THRESHOLD_DEG)[::-1]
    )[::-1]
    settled = np.flatnonzero(~above_threshold_later)
    return None if settled.size == 0 else float(time_s[settled[0]])


def orbital_period_s() -> float:
    """Return the two-body period associated with the initial orbit radius."""
    return float(
        2.0 * np.pi * np.sqrt(ORBIT_RADIUS_KM**3 / EarthConstants.mu_e)
    )


def save_plot(
    quaternion_results: ADCS.SimulationResults,
    vector_results: ADCS.SimulationResults,
) -> Path:
    """Save the compact multi-orbit alignment comparison as a PDF."""
    time_s = np.asarray(quaternion_results.first().time_s, dtype=float)
    period_s = orbital_period_s()
    time_orbits = time_s / period_s
    quaternion_error = quaternion_error_deg(quaternion_results)
    vector_error = vector_error_deg(vector_results)
    vector_settle_s = settling_time(time_s, vector_error)
    quaternion_settle_s = settling_time(time_s, quaternion_error)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)

    orbit_index = 0
    while orbit_index * period_s < SIM_DURATION_S:
        if orbit_index % 2 == 1:
            ax.axvspan(
                orbit_index,
                min(orbit_index + 1, time_orbits[-1]),
                color="#EEF1F4",
                zorder=0,
            )
        orbit_index += 1

    ax.axhline(
        ERROR_THRESHOLD_DEG,
        color="#7A8088",
        linewidth=0.7,
        linestyle="--",
        zorder=1,
    )
    ax.plot(
        time_orbits,
        quaternion_error,
        color=QUATERNION_COLOR,
        linewidth=1.5,
        label="Quaternion goal (full attitude)",
        zorder=2,
    )
    ax.plot(
        time_orbits,
        vector_error,
        color=VECTOR_COLOR,
        linewidth=1.8,
        label="Vector goal (roll free)",
        zorder=3,
    )

    vector_settle_text = (
        ">2 orbits"
        if vector_settle_s is None
        else f"{vector_settle_s / period_s:.2f} orbits"
    )
    quaternion_settle_text = (
        ">2 orbits"
        if quaternion_settle_s is None
        else f"{quaternion_settle_s / period_s:.2f} orbits"
    )
    ax.text(
        0.02,
        0.15,
        r"Settling below $10^\circ$" + "\n"
        + f"Vector: {vector_settle_text}\n"
        + f"Quaternion: {quaternion_settle_text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#30343B",
    )

    ax.set_title("3-MTQ Inertial Alignment with Lovera Control")
    ax.set_xlabel("Time [orbits]")
    ax.set_ylabel("Alignment error [deg]")
    ax.set_xlim(time_orbits[0], time_orbits[-1])
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    output_path = SCRIPT_DIR / "alignment_error.pdf"
    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    return output_path


def main() -> None:
    quaternion_goal = ADCS.goals.Fixed_Attitude_Goal(q_ref=Q_REFERENCE)
    vector_goal = ADCS.goals.ECI_Goal(eci_vector=VECTOR_REFERENCE)

    warnings.filterwarnings(
        "ignore",
        message="requested torque exceeds actuation limit",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in divide",
        category=RuntimeWarning,
        module="ADCS.controller.mtq_lovera",
    )
    print("Simulating full quaternion alignment...")
    quaternion_results = run_case(quaternion_goal)
    print("Simulating roll-free vector alignment...")
    vector_results = run_case(vector_goal)
    print(f"Saved {save_plot(quaternion_results, vector_results)}")


if __name__ == "__main__":
    main()
