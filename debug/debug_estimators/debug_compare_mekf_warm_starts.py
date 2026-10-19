"""Compare MEKF warm starts on a seeded scenario and save figures.

Run from any directory with
``python debug/debug_estimators/debug_compare_mekf_warm_starts.py``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.math_helpers import quaternion_angle


DT = 20.0
TF = 600.0
SEED = 2021
OUTPUT_DIR = ROOT / "docs/source/_static/release_notes"


def sensors():
    result = [ADCS.MTM(axis, noise=ADCS.Noise(std_noise=5.0e-8)) for axis in np.eye(3)]
    result += [ADCS.Gyro(axis, noise=ADCS.Noise(std_noise=5.0e-7)) for axis in np.eye(3)]
    result += [ADCS.SunPair(axis, efficiency=0.3, noise=ADCS.Noise(std_noise=1.0e-3)) for axis in np.eye(3)]
    return result


def initial_state():
    attitude = [np.cos(np.deg2rad(170.0)), 0.0, np.sin(np.deg2rad(170.0)), 0.0]
    return ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018], q=attitude,
        cov=np.diag([0.01**2] * 3 + [0.15**2] * 3),
        int_cov=np.diag([1.0e-16] * 3 + [1.0e-8] * 3),
    )


def run(warm_start=None):
    """Return time, attitude error, and the first attitude estimate's error."""
    np.random.seed(SEED)
    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    guess = warm_start if warm_start is not None else initial_state()
    estimator = ADCS.MEKF(estimated, guess, dt=DT)
    ephem = ADCS.Ephemeris()
    probe = ADCS.Orbital_State(
        ephem=ephem, J2000=0.22, R=[7000.0, 0.0, 0.0], V=[0.0, 7.5, 0.0],
    )
    sun = probe.S / np.linalg.norm(probe.S)
    velocity = np.cross(sun, [0.0, 0.0, 1.0])
    velocity = 7.5 * velocity / np.linalg.norm(velocity)
    os0 = ADCS.Orbital_State(ephem=ephem, J2000=0.22, R=7000.0 * sun, V=velocity)
    truth = ADCS.State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    result = ADCS.simulate(
        x=truth, satellite=satellite, est_satellite=estimated, estimator=estimator,
        os0=os0, dt=DT, tf=TF,
    ).first()
    error = np.rad2deg([
        quaternion_angle(real.q, estimate.q)
        for real, estimate in zip(result.state_hist, result.est_state_hist)
    ])
    guess_q = (estimator.warm_start_attitude.quaternion
               if warm_start is not None else guess.q)
    initial_error = np.rad2deg(quaternion_angle(truth.q, guess_q))
    return (np.insert(np.asarray(result.time_s), 0, 0.0),
            np.insert(error, 0, initial_error), initial_error)


def main():
    methods = (
        ("No warm start", None),
        ("TRIAD", ADCS.WarmStart(method="triad")),
        ("QUEST", ADCS.WarmStart(method="quest")),
        ("q-method", ADCS.WarmStart(method="q_method")),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, initial_errors, colors = [], [], []
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, start), color in zip(methods, palette):
        time_s, error_deg, initial_error = run(start)
        if np.any(error_deg <= 0):
            raise ValueError("Logarithmic attitude-error plot requires positive errors")
        ax.plot(time_s, error_deg, label=label, color=color)
        labels.append(label)
        initial_errors.append(initial_error)
        colors.append(color)
        print(f"{label}: initial attitude error {initial_error:.3f} deg")

    ax.set(xlabel="Time (s)", ylabel="Attitude error (deg, log scale)",
           title="MEKF with and without single-frame warm starts")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    bar_fig, bar_ax = plt.subplots(figsize=(8, 5))
    bars = bar_ax.bar(labels, initial_errors, color=colors)
    bar_ax.bar_label(bars, fmt="%.2f°", padding=3)
    bar_ax.set(ylabel="Initial attitude error (deg, log scale)",
               title="Initial MEKF attitude error: guess vs warm starts")
    bar_ax.set_yscale("log")
    bar_ax.set_ylim(0.05, max(initial_errors) * 1.5)
    bar_ax.grid(axis="y", alpha=0.3)
    bar_fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "0_2_1_mekf_warm_start.png", dpi=180)
    bar_fig.savefig(OUTPUT_DIR / "0_2_1_warm_start_initial_error.png", dpi=180)
    plt.close("all")


if __name__ == "__main__":
    main()
