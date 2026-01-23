import sys
import os
import numpy as np
from typing import Tuple
from typing import List
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.estimators.attitude_estimators import SRUAKF
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import GeometryFace, GeometryConfig, Drag_Disturbance, GG_Disturbance, SRP_Disturbance, DisturbanceMode
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.actuators import MTQ, Noise, Bias
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants

from ADCS.satellite_factory.actuators import create_cubewheel_smallplus_rw, create_isis_magnetorquer_board
from ADCS.satellite_factory.sensors import create_Clydespace_3U_array, create_ICM20948_IMU, create_isis_magnetometer

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window


def create_beavercube1_cubesat(estimated: bool = True):
    mass = 4
    COM = np.zeros(3)
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
    
    triple_bias = [Bias() for j in range(3)]
    triple_noise = [Noise() for j in range(3)]

    # Actuators
    mtqs: List[MTQ] = create_isis_magnetorquer_board(bias=triple_bias, estimate_bias=False)
    
    # Sensors
    mtms: List[MTM] = create_isis_magnetometer(estimate_bias=True)
    gyros: List[Gyro] = create_ICM20948_IMU(estimate_bias=True)
    solar_panel_1 = create_Clydespace_3U_array(axis=np.array([1, 0, 0]), bias=Bias(), estimate_bias=False)
    solar_panel_2 = create_Clydespace_3U_array(axis=np.array([0, 1, 0]), bias=Bias(), estimate_bias=False)
    suns: List[SunPair] = solar_panel_1+solar_panel_2

    # Disturbances
    geometry_faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2)]
    config = GeometryConfig(geometry_faces)
    gg_dist = [GG_Disturbance()]
    drag_dist = [Drag_Disturbance(config)]
    srp_dist = [SRP_Disturbance(config)]

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs)
    

def create_covariance_matrices(est_sat: EstimatedSatellite, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    mtm_bsr = 1e-9
    gyro_bsr = 0.0004*np.pi/180.0

    invJ = np.linalg.inv(est_sat.J_0)
    # Dynamics
    sigma_torque = 1e-4 
    Q_torque_continuous = np.eye(3) * sigma_torque**2
    Q_alpha = invJ @ Q_torque_continuous @ invJ.T
    Q_omega = Q_alpha * dt
    Q_att   = Q_alpha * (dt**3 / 3.0)
    Q_cross = Q_alpha * (dt**2 / 2.0)
    Q_dyn_block = np.block([
        [Q_omega, Q_cross],
        [Q_cross, Q_att]
    ])
    Q_mtm = 50 * np.eye(3) * (mtm_bsr)**2.0 * dt
    Q_gyro = np.eye(3) * (gyro_bsr)**2.0 * dt

    Q_est = block_diag(Q_dyn_block, Q_mtm, Q_gyro)
    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*3, np.eye(3)*(1e-8)**2.0, np.eye(3)*(0.2*np.pi/180.0)**2.0)
    
    return P_est, Q_est


def create_orbit(t0: float, tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = -7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if True:
        os = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os]*(dur+10)
        for j in range(dur):
            orbs[j] = os.copy()
            orbs[j].J2000 = os.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    return orb


def plot_results(time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist) -> None:
    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    plot_bias_comparison(time_hist, state_hist[:,7:10], est_state_hist[:,7:10], 
                        "Real vs Estimated MTM Bias", "T/s")
    plot_bias_comparison(time_hist, state_hist[:,10:13], est_state_hist[:,10:13], 
                        "Real vs Estimated Gyro Bias", "T/s")
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    create_close_all_button_window()


def main() -> None:
    np.random.seed(3)
    t0 = 0
    tf = 200
    dt = 1
    N = int((tf-t0)/dt)

    real_sat = create_beavercube1_cubesat()
    est_sat = create_beavercube1_cubesat(estimated=True)

    w0 = np.array([0, 0, 0])
    q0 = np.array([1, 0, 0, 0])
    x = np.concatenate([w0, q0])

    x_hat = np.zeros(13) #7 state, 3 MTM, 3 gyro
    x_hat[3] = 1

    P_est, Q_est = create_covariance_matrices(est_sat, dt)
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    orb = create_orbit(t0, tf, dt)
    os = orb.get_os(0.22)
    ukf = SRUAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, ukf.state_len))
    est_state_hist = np.nan*np.zeros((N, ukf.state_len))
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 8))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 8))
    u_hist = np.nan*np.zeros((N, 3))
    cov_hist: List[np.ndarray] = list()

    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)

    for step in tqdm(range(steps), desc="Simulating ukf"):
        u = np.zeros(3)

        dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os, dmode=dmode)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        x_hat = ukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        time_hist[ind] = t
        # real_mtq_biases = np.concatenate([mtq.bias.bias for mtq in real_sat.actuators if isinstance(mtq, MTQ)])
        real_gyro_biases = np.concatenate([gyro.bias.bias for gyro in real_sat.sensors if isinstance(gyro, Gyro)])
        real_mtm_biases = np.concatenate([mtm.bias.bias for mtm in real_sat.sensors if isinstance(mtm, MTM)])
        # real_sun_biases = np.concatenate([sun.bias.bias for sun in real_sat.sensors if isinstance(sun, SunPair)])
        # state_hist[ind,:] = np.concatenate([x, real_mtq_biases, real_mtm_biases, real_gyro_biases, real_sun_biases])
        state_hist[ind,:] = np.concatenate([x, real_mtm_biases, real_gyro_biases])
        est_state_hist[ind,:] = x_hat
        os_hist += [os]
        sensor_hist[ind,:] = noisy_sensor_readings
        clean_sensor_hist[ind,:] = clean_sensor_readings
        u_hist[ind,:] = u
        cov_hist += [ukf.x_hat.cov]

        # Propagate
        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    plot_results(time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist)



if __name__ == "__main__":
    main()