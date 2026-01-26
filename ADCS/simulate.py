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
    est_satellite: Optional[Satellite], 
    controller: Optional[Controller], 
    estimator: Optional[Attitude_Estimator],
    orbit_estimator: Optional[Orbit_Estimator],
    goal: Optional[Goal],
    os0: Orbital_State,
    dt: float = 1.0,
    tf: float = 500.0
) -> SimulationResults:
    if len(x) != satellite.state_len:
        raise ValueError(f"Initial state length {len(x)} does not match satellite state length {satellite.state_len}. It must be 7 + N_rw.")
    
    N = int(tf/dt)
    
    start_time = os0.J2000
    end_time = start_time + tf * TimeConstants.sec2cent
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True)

    u = np.zeros(satellite.control_len)

    if goal is None:
        goal = No_Goal()

    if est_satellite is not None and estimator is None:
        raise ValueError("Estimator must be provided if an estimated satellite is given.")
    
    if estimator is not None:
        if est_satellite is None:
            est_satellite = EstimatedSatellite.from_satellite(satellite)
        x_hat = np.empty(est_satellite.state_len)

    sim_results = SimulationResults()

    for k in tqdm(range(N), desc="Simulating ADCS", unit="step"):
        J2000 = start_time + k * dt * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        noisy_sensor_readings = satellite.sensor_readings(x=x, os=os)
        clean_sensor_readings = satellite.noiseless_sensor_readings(x=x, os=os)

        if orbit_estimator is not None:
            gps_readings = satellite.GPS_readings(x=x, os=os)
            os_hat: EstimatedOrbital_State = orbit_estimator.update(GPS_measurements=gps_readings, J2000=J2000)

        if estimator is not None:
            input_os = os_hat if os_hat is not None else os
            x_hat = estimator.update(u=u, sensors=noisy_sensor_readings, os=input_os)

        if controller is not None:
            input_os = os_hat if os_hat is not None else os
            input_x = x_hat if estimator is not None else x
            u = controller.find_u(x_hat=input_x, sens=noisy_sensor_readings, est_sat=est_satellite, os_hat=input_os, goal=goal)

        prev_os = os.copy()
        os = orb.get_os(J2000=start_time + (k+1) * dt * TimeConstants.sec2cent)
        out = solve_ivp(fun=satellite.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        sim_results.record(
            k=k,
            time_J2000=J2000,
            time_s=k*dt,
            os=os,
            est_os=os_hat if orbit_estimator is not None else None,
            os_cov=orbit_estimator.os_hat.P if orbit_estimator is not None else None,
            state=x,
            est_state=x_hat if estimator is not None else None,
            state_cov=estimator.x_hat.cov if estimator is not None else None,
            actuator_bias=np.array([act.bias.bias for act in satellite.actuators]) if len(satellite.actuators) > 0 else None,
            est_actuator_bias=x_hat[7:7+len(satellite.actuators)] if estimator is not None and len(satellite.actuators) > 0 else None,
            sensor_bias=np.array([sens.bias.bias for sens in satellite.sensors]) if len(satellite.sensors) > 0 else None,
            est_sensor_bias=x_hat[7+len(satellite.actuators):7+len(satellite.actuators)+len(satellite.sensors)] if estimator is not None and len(satellite.sensors) > 0 else None,
            clean_sensor_hist=clean_sensor_readings,
            sensor_hist=noisy_sensor_readings,
            control_hist=u
        )