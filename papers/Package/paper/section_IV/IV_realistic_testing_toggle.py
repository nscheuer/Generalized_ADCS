"""Progressively realistic single-run ADCS tests.

The four cases in :func:`IV_realistic_testing_toggle` deliberately add one
source of realism at a time:

1. perfect state knowledge and ideal dynamics;
2. noisy measurements, while retaining perfect state knowledge;
3. closed-loop state estimation;
4. environmental disturbance torques.

This is a small, repeatable example intended for Section IV, rather than a
Monte-Carlo campaign.  Set ``show_plots=False`` when using it from a test or
another script.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

import ADCS


EXPECTED_VERSION = "0.1.8"
OUTPUT_DIR = Path(__file__).with_name("outputs")
DT_S = 1.0
TF_S = 500.0
BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])
LEVEL_NAMES = (
    "Level 1: perfect knowledge",
    "Level 2: sensor noise",
    "Level 3: state estimation",
    "Level 4: disturbances",
)


def _satellite(*, noisy: bool = False, disturbances: list[Any] | None = None) -> ADCS.Satellite:
    """Build the plant used by every level."""
    mtqs = [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
    rws = [
        ADCS.RW(
            axis=BODY_BORESIGHT,
            max_torque=0.23e-3,
            J=5.7e-6,
            h=0.0,
            h_max=3.5e-3,
        )
    ]
    if noisy:
        gyro_noise = ADCS.Noise(std_noise=3.1623e-7)
        mtm_noise = ADCS.Noise(std_noise=5e-8)
        sensors = [ADCS.Gyro(axis, noise=gyro_noise) for axis in np.eye(3)]
        sensors += [ADCS.MTM(axis, noise=mtm_noise) for axis in np.eye(3)]
    else:
        sensors = [ADCS.Gyro(axis) for axis in np.eye(3)]
        sensors += [ADCS.MTM(axis) for axis in np.eye(3)]

    return ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([0.03, 0.03, 0.01]),
        actuators=mtqs + rws,
        sensors=sensors,
        disturbances=[] if disturbances is None else disturbances,
        boresight=BODY_BORESIGHT,
    )


def _controller(satellite: ADCS.Satellite) -> Any:
    return ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5e-5,
        d_gain=2e-3,
        c_gain=1e-3,
        h_target=np.zeros(3),
    )


def _initial_conditions() -> tuple[ADCS.State, ADCS.goals.Goal, ADCS.Orbital_State]:
    x_0 = ADCS.State.from_array(np.array([0.01, -0.02, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0]))
    goal = ADCS.goals.ECI_Goal(np.array([1.0, 0.0, 0.0]))
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )
    return x_0, goal, os0


def _estimator(
    est_satellite: ADCS.EstimatedSatellite, os0: ADCS.Orbital_State
) -> ADCS.SRUAKF:
    """Create the SRUAKF estimator provided by Generalized_ADCS 0.1.8."""
    # SRUAKF stores covariance in the reduced state space: angular velocity,
    # three attitude-error coordinates, and one reaction-wheel momentum.
    P0 = block_diag(np.eye(3) * 1e-4, np.eye(3) * 1e-2, np.array([[1e-8]]))
    Q = block_diag(np.eye(3) * 1e-10, np.eye(3) * 1e-8, np.array([[1e-10]]))
    x_hat = ADCS.EstimatorState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.zeros(1),
        cov=P0,
        int_cov=Q,
    )
    return ADCS.SRUAKF(
        est_sat=est_satellite,
        J2000=os0.J2000,
        x_hat=x_hat,
        P_hat=P0,
        Q_hat=Q,
        dt=DT_S,
        quat_as_vec=False,
        ephem=os0.ephem,
    )


def _plot_level(
    result: ADCS.SimulationResults, level: int, *, save_plot: bool = True
) -> None:
    """Use the appropriate plots and optionally save the generated figure."""
    estimated = level >= 3
    sources = ["real", "estimated"] if estimated else ["real"]
    target_modes = ["real_target", "real_est"] if estimated else ["real_target"]
    plots = [
        ADCS.plots.TargetPlot(modes=target_modes, title="Pointing error"),
        ADCS.plots.QuaternionPlotCombined(sources=sources, title="Attitude"),
        ADCS.plots.AngularVelocityPlotCombined(sources=sources, title="Angular velocity"),
        ADCS.plots.ControlPlotCombined(title="Control effort"),
    ]
    if level >= 2:
        plots.append(ADCS.plots.SensorsPlot(sources=["real", "clean"], title="Measurements"))
    ADCS.plot(
        result,
        *plots,
        layout=(2, 3),
        figsize=(12, 7),
        title=LEVEL_NAMES[level - 1],
    )
    if save_plot:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"iv_realistic_testing_level_{level}.png"
        plt.gcf().savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {path}")


def IV_realistic_testing_toggle(*, show_plots: bool = True) -> dict[str, ADCS.SimulationResults]:
    """Run the four progressively realistic single simulations.

    The returned mapping is useful for comparing results programmatically.  A
    fresh satellite, controller, and estimator is built for every level so
    mutable sensor, actuator, and filter state cannot leak between runs.
    """
    if ADCS.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"This paper script requires Generalized_ADCS=={EXPECTED_VERSION}; "
            f"found {ADCS.__version__!r}. Install papers/Package/requirements-package.txt."
        )

    x_0, goal, os0 = _initial_conditions()
    cases = (
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    )
    results: dict[str, ADCS.SimulationResults] = {}
    for level, (noisy, estimated, disturbed) in enumerate(cases, start=1):
        dists = [ADCS.disturbances.GG_Disturbance()] if disturbed else []
        sat = _satellite(noisy=noisy, disturbances=dists)
        est_sat = ADCS.EstimatedSatellite.from_satellite(sat) if estimated else None
        if est_sat is not None:
            # Generalized_ADCS 0.1.8's SRUAKF expects a non-empty actuator
            # covariance block. Keep this floor on the estimator copy only;
            # it does not add actuator noise to the simulated plant.
            for actuator in est_sat.actuators:
                actuator.noise = ADCS.Noise(std_noise=1e-10)
        ctrl = _controller(est_sat or ADCS.EstimatedSatellite.from_satellite(sat))
        estimator = _estimator(est_sat, os0) if estimated and est_sat is not None else None
        result = ADCS.simulate(
            x=deepcopy(x_0),
            satellite=sat,
            est_satellite=est_sat,
            controller=ctrl,
            estimator=estimator,
            goal=goal,
            os0=os0,
            dt=DT_S,
            tf=TF_S,
        )
        results[LEVEL_NAMES[level - 1]] = result
        _plot_level(result, level, save_plot=True)

    if show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    IV_realistic_testing_toggle()
