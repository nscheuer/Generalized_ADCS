__all__ = ["simulate"]

import numpy as np
from typing import Optional
from tqdm import tqdm
from scipy.integrate import solve_ivp

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller.controller import Controller
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize

from ADCS.helpers.simresults import SimulationResults

def simulate(
    x: np.ndarray,
    satellite: Satellite,
    est_satellite: Optional[Satellite] = None,
    controller: Optional[Controller] = None,
    estimator: Optional[Attitude_Estimator] = None,
    orbit_estimator: Optional[Orbit_Estimator] = None,
    goal: Optional[Goal] = None,
    os0: Orbital_State = None,
    dt: float = 1.0,
    tf: float = 500.0,
) -> SimulationResults:
    if len(x) != satellite.state_len:
        raise ValueError(
            f"Initial state length {len(x)} does not match satellite state length "
            f"{satellite.state_len}. It must be 7 + N_rw."
        )

    N = int(tf / dt)

    if goal is None:
        goal = No_Goal()

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

    sim_results = SimulationResults()

    for k in tqdm(range(N), desc="Simulating ADCS", unit="step"):
        J2000_k = start_time + k * dt * TimeConstants.sec2cent
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

        if controller is not None:
            u = controller.find_u(
                x_hat=x_for_ctrl,
                sens=y,
                est_sat=est_satellite,
                os_hat=os_for_gnc,
                goal=goal,
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

        sim_results.record(
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
            clean_sensor=y_clean,
            sensor=y,
            control=u,
        )

    return sim_results