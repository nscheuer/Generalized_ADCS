__all__ = ["simulate"]

import numpy as np
import time
from typing import Optional
from tqdm import tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import ADCS as ADCS

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.controller import Controller
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize

from ADCS.helpers.simresults import SimulationResults, RunResults
from ADCS.formation.satellite_agent import SatelliteAgent


def _supports_planning(controller: Controller) -> bool:
    """Return True if controller exposes plan-and-track style methods."""
    return callable(getattr(controller, "calculate_trajectory", None)) and callable(
        getattr(controller, "set_active_trajectory", None)
    )

def simulate(
    x: np.ndarray,
    satellite: Satellite,
    est_satellite: Optional[EstimatedSatellite] = None,
    controller: Optional[Controller] = None,
    estimator: Optional[Attitude_Estimator] = None,
    orbit_estimator: Optional[Orbit_Estimator] = None,
    goal: Optional[Goal | GoalList] = None,
    os0: Orbital_State = None,
    dt: float = 1.0,
    tf: float = 500.0,
) -> SimulationResults:
    r"""
    Run a time-domain simulation of the spacecraft Attitude Determination and Control
    System (ADCS), including dynamics propagation, sensor simulation, state estimation,
    orbit estimation, goal management, and control execution.

    This function advances the true satellite state forward in time using numerical
    integration, while optionally running attitude and orbit estimators and a control
    law. Goals may be specified as a single goal or a time-varying goal list. All
    simulation data are logged and returned as a :class:`~ADCS.helpers.simresults.SimulationResults`
    object.

    The simulation uses :func:`scipy.integrate.solve_ivp` with RK45 integration and
    propagates the orbit using :class:`~ADCS.orbits.orbit.Orbit`.

    :param x:
        Initial true satellite state vector. The length must match
        ``satellite.state_len`` and is expected to follow the satellite state
        convention (angular velocity, quaternion, reaction wheel states, etc.).
    :type x:
        numpy.ndarray

    :param satellite:
        The true satellite model, including dynamics, sensors, and actuators.
    :type satellite:
        :class:`~ADCS.satellite_hardware.satellite.Satellite`

    :param est_satellite:
        Estimated satellite model used by estimators and controllers. If ``None`` and
        either an estimator or controller is provided, it is constructed automatically
        from ``satellite``.
    :type est_satellite:
        :class:`~ADCS.satellite_hardware.satellite.EstimatedSatellite` or None

    :param controller:
        Control law used to compute actuator commands. If the controller is a
        :class:`~ADCS.controller.PlanAndTrackBase`, an initial trajectory is computed
        before simulation.
    :type controller:
        :class:`~ADCS.controller.Controller` or None

    :param estimator:
        Attitude estimator used to estimate the spacecraft state from sensor
        measurements.
    :type estimator:
        :class:`~ADCS.estimators.attitude_estimators.Attitude_Estimator` or None

    :param orbit_estimator:
        Orbit estimator used to estimate the orbital state from GPS measurements.
    :type orbit_estimator:
        :class:`~ADCS.estimators.orbit_estimators.Orbit_Estimator` or None

    :param goal:
        Desired attitude or pointing objective. This may be ``None`` (no goal),
        a single :class:`~ADCS.CONOPS.goals.Goal`, or a
        :class:`~ADCS.CONOPS.goallist.GoalList` defining time-varying goals.
    :type goal:
        :class:`~ADCS.CONOPS.goals.Goal`,
        :class:`~ADCS.CONOPS.goallist.GoalList`,
        or None

    :param os0:
        Initial orbital state at the start of the simulation.
    :type os0:
        :class:`~ADCS.orbits.orbital_state.Orbital_State`

    :param dt:
        Simulation time step in seconds.
    :type dt:
        float

    :param tf:
        Total simulation duration in seconds.
    :type tf:
        float

    :return:
        Container holding all recorded simulation data, including true and estimated
        states, controls, sensor readings, biases, and targets for the entire run.
    :rtype:
        :class:`~ADCS.helpers.simresults.SimulationResults`

    """
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length "
            f"{satellite.state_len}. It must be 7 + N_rw."
        )

    N = int(tf / dt)

    if goal is None:
        goal_list = GoalList({os0.J2000: No_Goal()})
    elif isinstance(goal, Goal):
        goal_list = GoalList({os0.J2000: goal})
    elif isinstance(goal, GoalList):
        goal_list = goal
    else:
        raise ValueError("goal must be None, a Goal, or a GoalList.")

    start_time = os0.J2000
    end_time = start_time + tf * TimeConstants.sec2cent
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)

    u = np.zeros(satellite.control_len)

    need_est_sat = (estimator is not None) or (controller is not None)
    if need_est_sat and est_satellite is None:
        est_satellite = EstimatedSatellite.from_satellite(satellite)

    x_hat = None
    if estimator is not None:
        x_hat = np.empty(est_satellite.state_len)

    os_hat = None

    if controller is not None and _supports_planning(controller):
        has_active_trajectory = getattr(controller, "active_trajectory", None) is not None
        if not has_active_trajectory:
            print("Calculating initial trajectory for Plan-and-Track controller...")
            trajectory = controller.calculate_trajectory(
                t_start=start_time,
                duration=tf,
                x_0=x,
                os_0=os0,
                goals=goal_list,
                verbose=False
            )

            if True:
                target_hist = []
                w_target_hist = []
                boresight_hist = []

                for t_j2000 in np.asarray(trajectory.times):
                    os_t = orb.get_os(J2000=float(t_j2000))
                    active_goal = goal_list.get_active_goal(float(t_j2000), time_units="centuries")
                    target_t, w_target_t = active_goal.to_ref(os_t)

                    target_hist.append(np.asarray(target_t, dtype=float).copy())
                    w_target_hist.append(np.asarray(w_target_t, dtype=float).copy())

                    boresight_vec = np.full(3, np.nan, dtype=float)
                    try:
                        b = est_satellite.get_boresight(active_goal.boresight_name)
                        boresight_vec = np.asarray(b, dtype=float).reshape(3)
                    except (AttributeError, KeyError, ValueError, TypeError):
                        pass
                    boresight_hist.append(boresight_vec)

                simresults = trajectory.to_simulation_results(
                    satellite,
                    target=np.asarray(target_hist),
                    w_target=np.asarray(w_target_hist),
                    boresight=np.asarray(boresight_hist),
                )
                ADCS.plot(
                    simresults,
                    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
                    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
                    ADCS.plots.TargetHistogram(bin_width=5.0),
                    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
                    layout=(2,2),
                    title="Open-Loop Planned Trajectory",
                )
                plt.show()

            controller.set_active_trajectory(trajectory)

    # The per-step loop body lives in SatelliteAgent so that single-satellite
    # simulation and multi-satellite (formation) simulation share exactly one
    # code path. simulate() owns only the orbit precompute + time grid here.
    agent = SatelliteAgent(
        x=x,
        satellite=satellite,
        est_satellite=est_satellite,
        controller=controller,
        estimator=estimator,
        orbit_estimator=orbit_estimator,
        goal_list=goal_list,
    )

    for k in tqdm(range(N), desc="Simulating ADCS", unit="step"):
        J2000_k = start_time + k * dt * TimeConstants.sec2cent
        os_k = orb.get_os(J2000=J2000_k)

        J2000_kp1 = start_time + (k + 1) * dt * TimeConstants.sec2cent
        os_kp1 = orb.get_os(J2000=J2000_kp1)

        agent.step(k, J2000_k, os_k, os_kp1)

    return SimulationResults(runs=[agent.results])