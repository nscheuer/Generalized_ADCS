__all__ = ["simulate"]

import numpy as np
from typing import Optional
from tqdm import tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import ADCS as ADCS

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller, PlanAndTrackBase
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize

from ADCS.helpers.simresults import SimulationResults, RunResults

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
    precompute_orbit: bool = True,
    progress: bool = True,
    log_mode: str = "full",
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

    :param precompute_orbit:
        If ``True``, precompute all orbital states on the simulation grid once and reuse
        them inside the time loop. This reduces repeated interpolation/lookups and
        improves runtime for most tutorials.
    :type precompute_orbit:
        bool

    :param progress:
        If ``True``, show a tqdm progress bar. Disable for slightly lower overhead in
        batched/headless runs.
    :type progress:
        bool

    :param log_mode:
        Controls per-step result logging detail. Supported values:

        - ``"full"``: record all telemetry fields (default, backward-compatible)
        - ``"minimal"``: record only core timelines/state/control/targets
        - ``"none"``: skip per-step logging entirely (maximum runtime throughput)

    :type log_mode:
        str

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

    if log_mode not in {"full", "minimal", "none"}:
        raise ValueError("log_mode must be one of: 'full', 'minimal', 'none'.")

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

    orbit_states = None
    if precompute_orbit:
        orbit_states = [
            orb.get_os(J2000=start_time + i * dt * TimeConstants.sec2cent)
            for i in range(N + 1)
        ]

    u = np.zeros(satellite.control_len)

    need_est_sat = (estimator is not None) or (controller is not None)
    if need_est_sat and est_satellite is None:
        est_satellite = EstimatedSatellite.from_satellite(satellite)

    x_hat = None
    if estimator is not None:
        x_hat = np.empty(est_satellite.state_len)

    os_hat = None

    if controller is not None and isinstance(controller, PlanAndTrackBase):
        print("Calculating initial trajectory for Plan-and-Track controller...")
        trajectory = controller.calculate_trajectory(
            t_start=start_time,
            duration=tf,
            x_0=x,
            os_0=os0,
            goals=goal_list,
            verbose=True
        )
        
        if True:
            active_goal = goal_list.get_active_goal(start_time, time_units="centuries")
            target, w_target = active_goal.to_ref(os0) 
            simresults = trajectory.to_simulation_results(satellite, target=target, w_target=w_target)
            ADCS.plot(
                simresults,
                ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
                ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
                ADCS.plots.TargetHistogram(bin_width=5.0),
                ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
                layout=(2,2),
                title="TRAJECTORY",
            )
            plt.show()

        controller.set_active_trajectory(trajectory)

    run_capsule = RunResults(satellite=satellite, est_satellite=est_satellite)

    step_iter = tqdm(range(N), desc="Simulating ADCS", unit="step") if progress else range(N)

    for k in step_iter:
        J2000_k = start_time + k * dt * TimeConstants.sec2cent

        if orbit_states is not None:
            os_k = orbit_states[k]
            os_kp1 = orbit_states[k + 1]
        else:
            os_k = orb.get_os(J2000=J2000_k)
            J2000_kp1 = start_time + (k + 1) * dt * TimeConstants.sec2cent
            os_kp1 = orb.get_os(J2000=J2000_kp1)

        y = satellite.sensor_readings(x=x, os=os_k)
        y_clean = satellite.noiseless_sensor_readings(x=x, os=os_k)

        if orbit_estimator is not None:
            gps = satellite.GPS_readings(x=x, os=os_k)
            os_hat = orbit_estimator.update(GPS_measurements=gps, J2000=J2000_k)
            os_for_gnc = os_hat if os_hat is not None else os_k
        else:
            os_hat = None
            os_for_gnc = os_k

        if estimator is not None:
            x_hat = estimator.update(u=u, sensors=y, os=os_for_gnc)
            x_for_ctrl = x_hat
        else:
            x_for_ctrl = x

        active_goal = goal_list.get_active_goal(J2000_k, time_units="centuries")

        if controller is not None:
            u = controller.find_u(
                x_hat=x_for_ctrl,
                sens=y,
                est_sat=est_satellite,
                os_hat=os_for_gnc,
                goal=active_goal,
            )
        else:
            u[:] = 0.0

        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os_k, os_kp1),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        target, w_target = active_goal.to_ref(os_for_gnc) 

        # Get the boresight vector for the current goal
        boresight_vec = None
        try:
            boresight_vec = est_satellite.get_boresight(active_goal.boresight_name)
        except (AttributeError, KeyError, ValueError, TypeError):
            pass

        est_act_bias_snapshot = None
        est_sens_bias_snapshot = None

        if estimator is not None and x_hat is not None and est_satellite is not None:
            # State layout per Attitude_Estimator doc:
            # [w(3), q(4), h_rw(n_rw), b_act(n_ab), b_sens(n_sb), theta_dist(n_dp)]
            n_rw = getattr(est_satellite, "number_RW", 0)
            n_ab = getattr(est_satellite, "act_bias_len", 0)
            n_sb = getattr(est_satellite, "att_sens_bias_len", 0)

            base = 7 + int(n_rw)
            ab0, ab1 = base, base + int(n_ab)
            sb0, sb1 = ab1, ab1 + int(n_sb)

            # Guard against unexpected shapes
            if len(x_hat) >= sb1:
                b_act_hat = np.asarray(x_hat[ab0:ab1], dtype=float).reshape(-1)
                b_sens_hat = np.asarray(x_hat[sb0:sb1], dtype=float).reshape(-1)

                # Actuator biases: slice b_act_hat according to per-actuator bias dimension
                act_parts = []
                ai = 0
                if getattr(satellite, "actuators", None):
                    for act in satellite.actuators:
                        # only include if bias exists (act.bias has __bool__)
                        if hasattr(act, "bias") and bool(act.bias):
                            # determine this actuator bias dimension from the true bias object
                            dim = int(np.atleast_1d(act.bias.bias).size)
                            act_parts.append(b_act_hat[ai:ai + dim].reshape(dim, 1) if dim == 1 else b_act_hat[ai:ai + dim])
                            ai += dim
                        else:
                            # keep placeholder for consistency with "real" formatting
                            act_parts.append(None)

                # If the estimator’s actuator-bias length doesn't match the sum of per-actuator dims,
                # fall back to storing the raw vector as a single entry.
                if len(act_parts) == 0:
                    est_act_bias_snapshot = None
                else:
                    # If our slicing didn't consume exactly the block, it's safer to store raw.
                    if ai != b_act_hat.size:
                        est_act_bias_snapshot = np.array([b_act_hat.copy()], dtype=object)
                    else:
                        est_act_bias_snapshot = np.array(act_parts, dtype=object)

                # Sensor biases: slice b_sens_hat according to per-sensor bias dimension
                sens_parts = []
                si = 0
                if getattr(satellite, "sensors", None):
                    for sens in satellite.sensors:
                        if hasattr(sens, "bias") and bool(sens.bias):
                            dim = int(np.atleast_1d(sens.bias.bias).size)
                            sens_parts.append(b_sens_hat[si:si + dim].reshape(dim, 1) if dim == 1 else b_sens_hat[si:si + dim])
                            si += dim
                        else:
                            sens_parts.append(None)

                if len(sens_parts) == 0:
                    est_sens_bias_snapshot = None
                else:
                    if si != b_sens_hat.size:
                        est_sens_bias_snapshot = np.array([b_sens_hat.copy()], dtype=object)
                    else:
                        est_sens_bias_snapshot = np.array(sens_parts, dtype=object)

        if log_mode == "full":
            run_capsule.record(
                k=k,
                time_J2000=J2000_k,
                time_s=k * dt,
                os=os_k,
                est_os=os_hat,
                os_cov=(getattr(getattr(orbit_estimator, "os_hat", None), "P", None)
                        if orbit_estimator is not None else None),
                state=x,
                est_state=x_hat,
                state_cov=(getattr(getattr(estimator, "x_hat", None), "cov", None)
                           if estimator is not None else None),
                actuator_bias=(
                    np.array([np.atleast_1d(act.bias.bias) for act in satellite.actuators], dtype=object)
                    if getattr(satellite, "actuators", None) else None
                ),
                sensor_bias=(
                    np.array([np.atleast_1d(sens.bias.bias) for sens in satellite.sensors], dtype=object)
                    if getattr(satellite, "sensors", None) else None
                ),
                est_actuator_bias=est_act_bias_snapshot,
                est_sensor_bias=est_sens_bias_snapshot,
                target=target,
                w_target=w_target,
                boresight=boresight_vec,
                clean_sensor=y_clean,
                sensor=y,
                control=u,
            )
        elif log_mode == "minimal":
            run_capsule.record(
                k=k,
                time_J2000=J2000_k,
                time_s=k * dt,
                state=x,
                est_state=x_hat,
                target=target,
                w_target=w_target,
                control=u,
            )

    return SimulationResults(runs=[run_capsule])