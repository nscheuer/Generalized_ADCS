import sys
import os
import numpy as np
from typing import List, Tuple
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from ADCS.controller import MTQ_w_RW, BDot
from ADCS.estimators.attitude_estimators.attitude_SRUAKF import SRUAKF
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import GeometryFace, GG_Disturbance, Drag_Disturbance, GeometryConfig
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize
from ADCS.state import EstimatedState, State
from ADCS.flight_software.single_core.ttc_single_core import TTC_Single_Core
from ADCS.flight_software.tasks.task import Task
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista

def create_satellite() -> Satellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_h0 = 1
    rw_hmax = 3.8
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_bias_mean = random_n_unit_vec(3)*np.random.uniform(1e-9, 1e-7)
    mtm_bsr = 1e-9
    mtm_bias = [Bias(bias=mtm_bias_mean[j], std_bias=mtm_bsr) for j in range(3)]
    mtms = [MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy()) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_bias_mean = np.array([0.002, 0.002, 0.002])
    gyro_bsr = 0.0004*np.pi/180.0
    gyro_bias = [Bias(bias=gyro_bias_mean[j], std_bias=gyro_bsr) for j in range(3)]
    gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy()) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    sun_bias_mean = np.array([0.05,0.09,-0.03])*sun_eff
    sun_bsr = 0.00001*sun_eff
    sun_bias = [Bias(bias=sun_bias_mean[j], std_bias=sun_bsr) for j in range(3)]
    suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy()) for j in range(3)]

    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    drag_dist = Drag_Disturbance(config=config)
    gg_dist = GG_Disturbance()
    dists = [drag_dist, gg_dist]

    real_sat_mass = 4.0
    real_sat_J = np.diagflat([3.4, 2.9, 1.3])
    real_sat = Satellite(mass=real_sat_mass, J_0=real_sat_J, actuators=mtqs+rws, sensors=mtms+gyros+suns, disturbances=dists)

    return real_sat
    
def create_estimated_satellite() -> EstimatedSatellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    est_mtqs = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_h0 = 1
    rw_hmax = 3.8
    est_rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_bsr = 1e-9
    mtm_bias = [Bias(bias=0.0, std_bias=mtm_bsr) for j in range(3)]
    est_mtms = [MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy(), estimate_bias=True) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_bsr = 0.0004*np.pi/180.0
    gyro_bias = [Bias(bias=0.0, std_bias=gyro_bsr) for j in range(3)]
    est_gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy(), estimate_bias=True) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    sun_bsr = 0.00001*sun_eff
    sun_bias = [Bias(bias=0.0, std_bias=sun_bsr) for j in range(3)]
    est_suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy(), estimate_bias=True) for j in range(3)]

    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    est_drag_dist = Drag_Disturbance(config=config)
    est_gg_dist = GG_Disturbance()
    est_dists = [est_drag_dist, est_gg_dist]

    # Satellite configuration
    est_sat_mass = 4.0
    est_sat_J = np.diagflat([3.4, 2.9, 1.3])
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_mtqs+est_rws, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

    return est_sat

def create_orbit(t0: float, tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = -7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if True:
        os = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os, end_time=end_time, dt=dt, zonal_J=2, fast=False)
    else:
        os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os]*(dur+10)
        for j in range(dur):
            orbs[j] = os.copy()
            orbs[j].J2000 = os.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    return orb

def create_matrices(est_sat: EstimatedSatellite, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    mtm_bsr = 1e-9
    gyro_bsr = 0.0004*np.pi/180.0
    sun_bsr = 0.00001
    
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
    # Biases
    mult_mtm = 1
    mult_sun = 10.0
    Q_mtm  = np.eye(3) * (mtm_bsr * mult_mtm)**2.0 * dt
    Q_gyro = np.eye(3) * (gyro_bsr)**2.0 * dt
    Q_sun  = np.eye(3) * (sun_bsr * mult_sun)**2.0 * dt
    Q_est = block_diag(Q_dyn_block, Q_mtm, Q_gyro, Q_sun)

    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*3, 0.001*np.eye(3)*mtm_bsr**2.0, np.eye(3)*1000*gyro_bsr**2.0, np.eye(3)*100*sun_bsr**2.0)

    return P_est, Q_est

def operating_system_task(t, memory):
    memory["log"].append(f"{t:.2f}: Operating System")

    # Switching control modes
    Pmax = np.max(memory["ESTIMATOR"].x_hat.cov)

    LOW_THRESH  = 1e-5   # enter MTQ_W_RW
    HIGH_THRESH = 5e-5   # fall back to BDOT

    if memory["MODE_control"] == "MODE_BDOT":
        if Pmax < LOW_THRESH:
            memory["MODE_control"] = "MODE_MTQ_W_RW"
            print("Switched to BDot")

    elif memory["MODE_control"] == "MODE_MTQ_W_RW":
        if Pmax > HIGH_THRESH:
            memory["MODE_control"] = "MODE_BDOT"
            print("Switched to MTQ_W_RW")

def estimator_task(t, memory):
    memory["log"].append(f"{t:.2f}: EST")
    ukf = memory["ESTIMATOR"]
    u = memory["control_u"]
    sensors = memory["sensor_readings"]
    os = memory["orbital_state"]

    memory["x_hat"] = ukf.update(u=u, sensors=sensors, os=os)

def controller_task(t, memory):
    memory["log"].append(f"{t:.2f}: CTRL")

    x_hat = memory["x_hat"]
    sens = memory["sensor_readings"]
    est_sat = memory["ESTIMATOR"].est_sat
    os_hat = memory["orbital_state"]

    if memory["MODE_control"] == "MODE_BDOT":
        memory["control_u"] = memory["CONTROL_BDOT"].find_u(x_hat, sens, est_sat, os_hat)
    elif memory["MODE_control"] == "MODE_MTQ_W_RW":
        memory["control_u"] = memory["CONTROL_MTQ_W_RW"].find_u(x_hat, sens, est_sat, os_hat)
    else:
        memory["control_u"] = np.zeros(6)

def sensor_task(t, memory):
    memory["log"].append(f"{t:.2f}: SENS")
    memory["sensor_readings"] = memory["sensor_readings"]


def main():
    np.random.seed(1)
    t0, tf = 0.0, 50.0
    dt = 0.1
    N = int((tf-t0)/dt)

    orb = create_orbit(t0, tf, dt=1)
    real_sat = create_satellite()
    est_sat = create_estimated_satellite()
    P_est, Q_est = create_matrices(est_sat, dt=10)

    x = State(w=np.zeros(3), q=[1, 0, 0, 0], h=np.ones(3))
    x_hat = EstimatedState(w=np.zeros(3), q=[1, 0, 0, 0], h=np.zeros(3), sens_bias=np.zeros(9))
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    ukf = SRUAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=10, cross_term=True, quat_as_vec=False)

    bdot = BDot(est_sat=est_sat, gain=100)
    mtq_w_rw = MTQ_w_RW(est_sat=real_sat, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.array([0, 0, 0]))

    memory = {
        "log": [],
        "sensor_readings": np.zeros(9),
        "control_u": np.zeros(3),
        "orbital_state": [],
        "x_hat": [],
        "ESTIMATOR": ukf,
        "CONTROL_BDOT": bdot,
        "CONTROL_MTQ_W_RW": mtq_w_rw,
        "MODE_control": "MODE_BDOT",
    }

    tasks = [
        Task("os", operating_system_task, rate_hz=10.0, wcet=0.001, priority=0),
        Task("sensor", sensor_task, rate_hz=5.0, wcet=0.001, priority=1),
        Task("estimator", estimator_task, rate_hz=0.1, wcet=0.020, priority=1),
        Task("controller", controller_task, rate_hz=0.1, wcet=0.002, priority=2),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory, debug=False)

    time_hist = np.nan*np.zeros(N)
    state_hist: List[State] = []
    est_state_hist: List[EstimatedState] = []
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    u_hist = np.nan*np.zeros((N, 6))
    real_mtm_bias_hist = np.nan*np.zeros((N, 3))
    est_mtm_bias_hist = np.nan*np.zeros((N, 3))
    real_gyro_bias_hist = np.nan*np.zeros((N, 3))
    est_gyro_bias_hist = np.nan*np.zeros((N, 3))
    real_sun_bias_hist = np.nan*np.zeros((N, 3))
    est_sun_bias_hist = np.nan*np.zeros((N, 3))
    cov_hist: List[np.ndarray] = list()

    t = t0
    ind = 0
    steps = int((tf-t0)/dt)
    for step in tqdm(range(steps), desc="Simulating TTC_Single_Core"):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000)
        core.memory["orbital_state"] = os

        dmode = ErrorMode(
            add_bias=True, add_noise=True,
            update_bias=True, update_noise=True
        )
        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os, dmode=dmode)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        memory["sensor_readings"] = noisy_sensor_readings

        core.step()
        u = core.memory["control_u"]

        time_hist[ind] = t
        real_gyro_biases = np.concatenate([gyro.bias.bias for gyro in real_sat.sensors if isinstance(gyro, Gyro)])
        real_mtm_biases = np.concatenate([mtm.bias.bias for mtm in real_sat.sensors if isinstance(mtm, MTM)])
        real_sun_biases = np.concatenate([sun.bias.bias for sun in real_sat.sensors if isinstance(sun, SunPair)])
        state_hist.append(x.copy())
        est_state_hist.append(memory["estimator"].x_hat.copy())
        real_mtm_bias_hist[ind,:] = real_mtm_biases
        est_mtm_bias_hist[ind,:] = memory["estimator"].x_hat.sens_bias[0:3]
        real_gyro_bias_hist[ind,:] = real_gyro_biases
        est_gyro_bias_hist[ind,:] = memory["estimator"].x_hat.sens_bias[3:6]
        real_sun_bias_hist[ind,:] = real_sun_biases
        est_sun_bias_hist[ind,:] = memory["estimator"].x_hat.sens_bias[6:9]
        os_hist += [core.memory["orbital_state"]]
        sensor_hist[ind,:] = core.memory["sensor_readings"]
        clean_sensor_hist[ind,:] = clean_sensor_readings
        u_hist[ind,:] = core.memory["control_u"]
        cov_hist += [ukf.x_hat.cov]

        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)
        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(), method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = State.from_array(out.y[:, -1])
        x = x.normalized()

    
    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    plot_bias_comparison(time_hist, real_gyro_bias_hist, est_gyro_bias_hist,
                        "Real vs Estimated Gyroscope Bias", "rad/s")
    plot_bias_comparison(time_hist, real_mtm_bias_hist, est_mtm_bias_hist,
                        "Real vs Estimated MTM Bias", "T/s")
    plot_bias_comparison(time_hist, real_sun_bias_hist, est_sun_bias_hist,
                        "Real vs Estimated Sun Bias", "W/s")
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    animate_orbit_pyvista(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist)
    create_close_all_button_window()



if __name__ == "__main__":
    main()
