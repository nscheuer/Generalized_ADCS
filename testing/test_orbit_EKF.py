import sys
import os
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.sensors import GPS
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.estimators.orbit_estimators import Orbit_EKF
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode

from plotting.plot_orbit_estimator import plot_gps_error
from plotting.close_all_plots import create_close_all_button_window

def run_orbit_ekf(verbose: bool = False, tf: float = 1000, dt: float = 10) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ephem = Ephemeris()
    t0 = 0
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time   = 0.22 + (tf - t0) * TimeConstants.sec2cent

    # Initial true orbit
    R = 7000 * np.array([0, -np.sqrt(2)/2, -np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)

    # True orbit generator
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)

    # -----------------------------
    # 2. Create GPS sensor models
    # -----------------------------
    gps_noise = Noise(
        noise=np.zeros(6),
        std_noise=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])   # km, km/s
    )

    # One GPS on the real "satellite"
    real_gps = GPS(noise=gps_noise.copy())

    # One GPS model for the estimator
    est_gps = GPS(noise=gps_noise.copy())

    # Small dummy satellite for estimator (only used to call GPS_readings)
    est_sat = Satellite(sensors=[est_gps])

    # -----------------------------
    # 3. Create Orbit Estimator
    # -----------------------------
    P0 = np.diag([500**2.0, 500**2.0, 500**2.0, 0.5**2.0, 0.5**2.0, 0.5**2.0])    # initial covariance
    Q0 = np.diag([1, 1, 1, 10, 10, 10])

    os_hat0 = Orbital_State(
        ephem=ephem,
        J2000=start_time,
        R=np.array([7000, 7000, 0]),
        V=np.array([0, 0, 8])
    )

    orbit_ekf = Orbit_EKF(
        est_sat=est_sat,
        J2000=start_time,
        os_hat=os_hat0,   # initial estimate
        P_hat=P0,
        Q_hat=Q0,
        dt=dt
    )

    # -----------------------------
    # 4. Logging Arrays
    # -----------------------------
    N = int((tf - t0)/dt)
    time_hist = np.zeros(N)
    true_hist = np.zeros((N,6))
    est_hist  = np.zeros((N,6))
    cov_hist  = []

    # -----------------------------
    # 5. Main Loop
    # -----------------------------
    t = 0
    for k in range(N):

        # True orbital state
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os_true = orb.get_os(J2000=J2000)

        # True GPS reading (no noise)
        # GPS does not really care about x
        gps_clean = est_gps.clean_reading(x=None, os=os_true)

        # Noisy measurement for update
        dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
        gps_meas = est_gps.reading(x=None, os=os_true, dmode=dmode)

        # EKF update
        est_os: EstimatedOrbital_State = orbit_ekf.update(
            GPS_measurements=[gps_meas],
            J2000=J2000
        )

        if verbose:
            print("Real Position: ", os_true.R)
            print("Estimated Position: ", est_os.os.R)
            print("Position Delta: ", np.abs(os_true.R - est_os.os.R))
            print("Real Velocity: ", os_true.V)
            print("Estimated Velocity: ", est_os.os.V)
            print("Velocity Delta: ", np.abs(os_true.V - est_os.os.V))
            print("")
            print("P: ", orbit_ekf.os_hat.P)
            print("Q: ", orbit_ekf.os_hat.Q)
            print("")

        # Store logs
        time_hist[k]     = t
        true_hist[k,:]   = np.hstack([os_true.R, os_true.V])
        est_hist[k,:]    = np.hstack([est_os.os.R, est_os.os.V])
        cov_hist.append(est_os.P.copy())

        # Next step
        t += dt

    print("Final Position Error: ", np.abs(os_true.R - est_os.os.R))
    print("Final Velocity Error: ", np.abs(os_true.V - est_os.os.V))
    return time_hist, true_hist, est_hist, cov_hist

def plot_orbit_ekf(verbose: bool, tf: float, dt: float) -> None:
    (time_hist, true_hist, est_hist, cov_hist) = run_orbit_ekf(verbose, tf, dt)

    plot_gps_error(time_hist, true_hist, est_hist)
    create_close_all_button_window()

if __name__ == "__main__":
    plot_orbit_ekf(verbose=False, tf=500, dt=5)