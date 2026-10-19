"""Compare four attitude filters on the same seeded scenario and save figures.

Run from any directory with ``python debug/debug_estimators/debug_compare_filters.py``.
The timing figure measures whole-simulation wall time per step after a warm-up
run, so it includes common orbit and plant work as well as estimator work.
"""

from pathlib import Path
import sys
from time import perf_counter

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


def initial_state(filter_type):
    attitude = [np.cos(np.deg2rad(50.0)), 0.0, np.sin(np.deg2rad(50.0)), 0.0]
    if filter_type is ADCS.EKF:
        return ADCS.EstimatorState(
            w=[0.0012, 0.0008, -0.0018], q=attitude,
            cov=np.diag([0.01**2] * 3 + [0.15**2] * 4),
            int_cov=np.diag([1.0e-16] * 3 + [0.0, 1.0e-8, 1.0e-8, 1.0e-8]),
        )
    return ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018], q=attitude,
        cov=np.diag([0.01**2] * 3 + [0.15**2] * 3),
        int_cov=np.diag([1.0e-16] * 3 + [1.0e-8] * 3),
    )


def run(filter_type):
    """Return time, attitude error, and wall time per simulated step."""
    np.random.seed(SEED)
    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    guess = initial_state(filter_type)
    estimator = filter_type(estimated, guess, dt=DT)
    ephem = ADCS.Ephemeris()
    probe = ADCS.Orbital_State(
        ephem=ephem, J2000=0.22, R=[7000.0, 0.0, 0.0], V=[0.0, 7.5, 0.0],
    )
    sun = probe.S / np.linalg.norm(probe.S)
    velocity = np.cross(sun, [0.0, 0.0, 1.0])
    velocity = 7.5 * velocity / np.linalg.norm(velocity)
    os0 = ADCS.Orbital_State(ephem=ephem, J2000=0.22, R=7000.0 * sun, V=velocity)
    truth = ADCS.State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    start = perf_counter()
    result = ADCS.simulate(
        x=truth, satellite=satellite, est_satellite=estimated, estimator=estimator,
        os0=os0, dt=DT, tf=TF,
    ).first()
    seconds_per_step = (perf_counter() - start) / len(result.time_s)
    error = np.rad2deg([
        quaternion_angle(real.q, estimate.q)
        for real, estimate in zip(result.state_hist, result.est_state_hist)
    ])
    initial_error = np.rad2deg(quaternion_angle(truth.q, guess.q))
    return (np.insert(np.asarray(result.time_s), 0, 0.0),
            np.insert(error, 0, initial_error), seconds_per_step)


def main():
    filters = (("EKF", ADCS.EKF), ("MEKF", ADCS.MEKF),
               ("UKF", ADCS.UKF), ("SRUKF", ADCS.SRUKF))
    colors = dict(zip((name for name, _ in filters), plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    # Prime JIT kernels and caches before comparing wall times. The measured
    # run reseeds the scenario and constructs fresh satellites and estimators.
    for _, filter_type in filters:
        run(filter_type)

    fig, ax = plt.subplots(figsize=(9, 5))
    tradeoff_fig, tradeoff_ax = plt.subplots(figsize=(8, 5))
    for name, filter_type in filters:
        time_s, error_deg, seconds_per_step = run(filter_type)
        error_at_100_s = float(np.interp(100.0, time_s[1:], error_deg[1:]))
        ax.plot(time_s, error_deg, label=name, color=colors[name])
        tradeoff_ax.scatter(seconds_per_step * 1e3, error_at_100_s,
                            s=110, color=colors[name], label=name)
        offset = (-40, 7) if name == "UKF" else (7, 5)
        tradeoff_ax.annotate(name, (seconds_per_step * 1e3, error_at_100_s),
                             xytext=offset, textcoords="offset points")
        print(f"{name}: {seconds_per_step * 1e3:.2f} ms/step; "
              f"{error_at_100_s:.3f} deg at 100 s")

    ax.set(xlabel="Time (s)", ylabel="Attitude error (deg)",
           title="EKF, MEKF, UKF and SRUKF on the same scenario")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    tradeoff_ax.set(xlabel="Simulation wall time per step (ms)",
                    ylabel="Attitude error at 100 s (deg)",
                    title="Filter speed and convergence after 100 s")
    tradeoff_ax.grid(True, alpha=0.35)
    tradeoff_fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "0_2_1_filter_comparison.png", dpi=180)
    tradeoff_fig.savefig(OUTPUT_DIR / "0_2_1_filter_tradeoff.png", dpi=180)
    plt.close("all")


if __name__ == "__main__":
    main()
