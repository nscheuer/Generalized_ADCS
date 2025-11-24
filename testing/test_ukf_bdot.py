import sys
import os
import numpy as np
import numdifftools as nd
import pytest
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union
from scipy.stats import kstest, ks_2samp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from asciichartpy import plot
import time
from tqdm import tqdm

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, norm, normalize, limit
from ADCS.helpers.math_constants import MathConstants
from ADCS.estimators.attitude_SRUKF import SRUKF
from ADCS.estimators.attitude_UKF import UKF

def bdot(B_now: np.ndarray, B_prev: np.ndarray, mtq_max_torque: float) -> np.ndarray:
    ud = -1e8*(B_now - B_prev)
    control = limit(ud, mtq_max_torque)
    return control

def test_ukf_bdot(verbose: bool = False, tf: float = 500, dt: float = 1, t_start_bdot: float = 300, real_orbit: bool = False) -> Union[np.ndarray, np.ndarray, np.ndarray, List[Orbital_State], List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray]]:
    np.random.seed(1)

    t0 = 0

    N = int((tf-t0)/dt)
    np.set_printoptions(precision=6)

    ## REAL SATELLITE
    # Actuators: Magnetorquers
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]

    # Sensors: Magnetometers
    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtms = [MTM(axis=j, noise=mtm_noise) for j in MathConstants.unitvecs]

    # Sensors: Gyroscopes
    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyros = [Gyro(axis=j, noise=gyro_noise) for j in MathConstants.unitvecs]

    # Sensors: SunPair
    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    suns = [SunPair(axis=j, efficiency=sun_eff, noise=sun_noise) for j in MathConstants.unitvecs]

    # Disturbance: Drag
    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    drag_dist = Drag_Disturbance(config=config)

    # Disturbance: Gravity Gradient
    gg_dist = GG_Disturbance()

    dists = [drag_dist, gg_dist]

    # Satellite configuration
    real_sat_mass = 4.0
    real_sat_J = np.diagflat([3.4, 2.9, 1.3])
    real_sat = Satellite(mass=real_sat_mass, J_0=real_sat_J, actuators=acts, sensors=mtms+gyros+suns, disturbances=dists)

    # Initial State
    w0 = random_n_unit_vec(3)*np.random.uniform(0, 0.1)*np.pi/180.0
    w0 = np.array([0, 0, 0])
    print(w0)

    q0 = random_n_unit_vec(4)
    print(q0)

    x = np.concatenate([w0, q0])
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if real_orbit:
        # Real Orbit Generation
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=1e-7)
        dur = int((tf-t0)/dt)+10
        orbs = [os0]*(dur+10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    ## ESTIMATED SATELLITE
    # Actuators
    est_acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]
    # Sensors
    est_mtms = [MTM(axis=j, noise=mtm_noise) for j in MathConstants.unitvecs]
    est_gyros = [Gyro(axis=j, noise=gyro_noise) for j in MathConstants.unitvecs]
    est_suns = [SunPair(axis=j, efficiency=sun_eff, noise=sun_noise) for j in MathConstants.unitvecs]
    # Disturbances
    est_drag_dist = Drag_Disturbance(config=config)
    est_gg_dist = GG_Disturbance()
    est_dists = [est_drag_dist, est_gg_dist]

    # Satellite configuration
    est_sat_mass = 4.0
    est_sat_J = np.diagflat([3.4, 2.9, 1.3])
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_acts, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

    # Initial Estimated State
    x_hat = np.zeros(7)
    x_hat[3] = 1
    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*3)
    Q_est = block_diag(np.eye(3)*(1e-4)**2.0, 1e-4*np.eye(3))

    ## Build Estimator
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    srukf = UKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    # Create history vectors
    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, srukf.state_len))
    est_state_hist = np.nan*np.zeros((N, srukf.state_len))
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    u_hist = np.nan*np.zeros((N, len(acts)))
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)
    os = orb.get_os(J2000=0.22)
    B_prev = np.array(real_sat.mtm_readings(x=x, os=os))
    B_now = B_prev
    
    for step in tqdm(range(steps), desc="Simulating UKF"):
        # One Step Propagation
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        B_now = np.array(real_sat.mtm_readings(x=x, os=os))
        if t < t_start_bdot:
            u = np.zeros(len(acts))
        else:
            u = bdot(B_now=B_now, B_prev=B_prev, mtq_max_torque=mtq_max_torque)
        x_hat = srukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        if verbose:
            # Full State Debug
            print("Real State ", x)
            print("Estimated State ", x_hat)

            # Attitude Debug
            quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat[3:7], x[3:7]), -1, 1)**2.0)
            print("Attitude Error (Degrees) ", quaternion_error_deg)
            angular_velocity_error = norm(x_hat[0:3] - x[0:3])*180.0/np.pi
            print("Angular Velocity Error ", angular_velocity_error)
            diagonal_covariances = np.diagonal(srukf.x_hat.cov)
            print("Attitude Covariance ", diagonal_covariances[3:6])
            print("Angular Velocity Covariance ", diagonal_covariances[0:3])
            print("")

        # Save Information for Plotting
        time_hist[ind] = t
        state_hist[ind,:] = x
        est_state_hist[ind,:] = x_hat
        os_hist += [os]
        sensor_hist[ind,:] = noisy_sensor_readings
        clean_sensor_hist[ind,:] = clean_sensor_readings
        u_hist[ind,:] = u
        cov_hist += [srukf.x_hat.cov]

        # Propagate
        ind += 1
        t += dt
        B_prev = B_now
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist


def plot_test_ukf_bdot(verbose: bool = False, tf: float = 500, dt: float = 1, t_start_bdot: float = 300, real_orbit: bool = False) -> None:
    (time_hist, state_hist, est_state_hist, os_hist,
     sensor_hist, clean_sensor_hist, u_hist, cov_hist) = test_ukf_bdot(
         verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    quat_err = np.zeros_like(time_hist)
    omega_err = np.zeros_like(time_hist)

    # === Convert quaternions to Euler angles ===
    def quat_to_euler(q):
        R = rot_mat(q)
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = -np.arcsin(R[2,0])
        yaw = np.arctan2(R[1,0], R[0,0])
        return np.array([roll, pitch, yaw]) * 180/np.pi

    euler_real = np.array([quat_to_euler(q) for q in state_hist[:, 3:7]])
    euler_est  = np.array([quat_to_euler(q) for q in est_state_hist[:, 3:7]])

    # === Compute errors ===
    for i in range(len(time_hist)):
        q_hat = est_state_hist[i, 3:7]
        q = state_hist[i, 3:7]
        qdot = np.clip(np.dot(q_hat, q), -1.0, 1.0)
        quat_err[i] = (180.0/np.pi) * np.arccos(-1 + 2 * qdot**2.0)
        omega_err[i] = norm(est_state_hist[i, 0:3] - state_hist[i, 0:3]) * 180.0/np.pi

    # ========= Real vs Estimated State =========
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    state_labels = ["ω₁", "ω₂", "ω₃"]
    euler_labels = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]

    # Angular velocity
    for i in range(3):
        axs[i].plot(time_hist, state_hist[:, i], label="Real")
        axs[i].plot(time_hist, est_state_hist[:, i], "--", label="Estimated")
        axs[i].set_title(state_labels[i])
        axs[i].grid(True)

    # Euler angles
    for i in range(3):
        axs[i+3].plot(time_hist, euler_real[:, i], label="Real")
        axs[i+3].plot(time_hist, euler_est[:, i], "--", label="Estimated")
        axs[i+3].set_title(euler_labels[i])
        axs[i+3].grid(True)
        axs[i+3].set_xlabel("Time [s]")

    axs[0].legend()
    fig.suptitle("Real vs Estimated States (ω and Euler Angles)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ============== Error Plots + Sunlit ==============
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    # Quaternion error
    ax1.plot(time_hist, quat_err, label="Quaternion Error")
    ax1.set_ylabel("Quat Err [deg]")
    ax1.grid(True)

    # Angular velocity error
    ax2.plot(time_hist, omega_err, label="Angular Velocity Error")
    ax2.set_ylabel("ω Error [deg/s]")
    ax2.grid(True)

    # Sunlit state (0/1)
    sunlit = np.array([os.is_sunlit() for os in os_hist]).astype(int)
    ax3.step(time_hist, sunlit, where="post", color="orange")
    ax3.set_ylim([-0.2, 1.2])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Dark", "Sunlit"])
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Sun")
    ax3.grid(True)

    fig.suptitle("Quaternion Error, Angular Velocity Error, and Sunlight State")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig3, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    axs = axs.flatten()

    # SENSOR PLOTS

    sensor_names = ["MTM X", "MTM Y", "MTM Z",
                    "GYR X", "GYR Y", "GYR Z",
                    "SUN X", "SUN Y", "SUN Z"]

    for i in range(9):
        axs[i].plot(time_hist, sensor_hist[:, i], label="Measured")
        axs[i].plot(time_hist, clean_sensor_hist[:, i], '--', label="Clean")
        axs[i].set_title(sensor_names[i])
        axs[i].grid(True)
        if i >= 6:
            axs[i].set_xlabel("Time [s]")

    axs[0].legend()
    fig3.suptitle("Measured Sensor Readings vs Clean Sensor Values")
    fig3.tight_layout(rect=[0, 0, 1, 0.96])


    # ============== CONTROL EFFORT PLOT ==============
    fig_u, ax_u = plt.subplots(figsize=(10, 4))

    ax_u.plot(time_hist, u_hist, label="Control Effort u")
    ax_u.set_title("Control Effort Over Time")
    ax_u.set_xlabel("Time [s]")
    ax_u.set_ylabel("Control Effort")
    ax_u.grid(True)
    ax_u.legend()

    fig_u.tight_layout()

    # ============== 3D ANIMATION ==============
    body_axes = np.eye(3)

    fig2 = plt.figure(figsize=(9, 9))
    ax = fig2.add_subplot(111, projection="3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Attitude + Magnetic Field + Sun Vector")

    # --- Body axes ---
    true_lines = [ax.plot([], [], [], lw=2)[0] for _ in range(3)]
    est_lines  = [ax.plot([], [], [], lw=1, linestyle="--")[0] for _ in range(3)]

    # --- Arrows ---
    B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', linewidth=2)
    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='yellow', linewidth=2)

    q_true_hist = state_hist[:, 3:7]
    q_est_hist  = est_state_hist[:, 3:7]

    # Animation state variables
    frame_index = [0]
    play_state = [True]
    speed_factor = [1.0]   # 1× by default

    def init_anim():
        nonlocal B_arrow, S_arrow  # MUST be first line

        # Clear body axis lines
        for ln in true_lines + est_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])

        # Reset arrows
        B_arrow.remove()
        S_arrow.remove()
        B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', linewidth=2)
        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='yellow', linewidth=2)

        return true_lines + est_lines + [B_arrow, S_arrow]


    def update_anim(_):
        nonlocal B_arrow, S_arrow  # MUST be first line

        # If paused, do nothing
        if not play_state[0]:
            return true_lines + est_lines + [B_arrow, S_arrow]

        # Advance frame index
        frame_index[0] = (frame_index[0] + speed_factor[0]) % len(time_hist)
        i = int(frame_index[0])

        Rt = rot_mat(q_true_hist[i])
        Re = rot_mat(q_est_hist[i])

        true_ax = Rt @ body_axes
        est_ax  = Re @ body_axes

        # Body axes updates
        for k in range(3):
            true_lines[k].set_data([0, true_ax[0, k]], [0, true_ax[1, k]])
            true_lines[k].set_3d_properties([0, true_ax[2, k]])

            est_lines[k].set_data([0, est_ax[0, k]], [0, est_ax[1, k]])
            est_lines[k].set_3d_properties([0, est_ax[2, k]])

        # Update B vector
        B_arrow.remove()
        B = os_hist[i].B / np.linalg.norm(os_hist[i].B)
        B_arrow = ax.quiver(0, 0, 0, B[0], B[1], B[2], color='magenta', linewidth=2)

        # Update sun vector depending on eclipse
        S_arrow.remove()
        if os_hist[i].is_sunlit():
            S = os_hist[i].S / np.linalg.norm(os_hist[i].S)
            S_arrow = ax.quiver(0, 0, 0, S[0], S[1], S[2], color='yellow', linewidth=2)
        else:
            # invisible arrow
            S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='yellow', linewidth=0)

        return true_lines + est_lines + [B_arrow, S_arrow]

    ani = FuncAnimation(fig2, update_anim, init_func=init_anim, interval=50, blit=False)

    # ----- UI CONTROLS -----
    from matplotlib.widgets import Button, RadioButtons

    # Pause / Play button
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")

    def toggle_play(event):
        play_state[0] = not play_state[0]
    btn_pause.on_clicked(toggle_play)

    # Speed selector
    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(ax_speed, ("0.25×", "0.5×", "1×", "2×", "4×"), active=2)

    def set_speed(label):
        mapping = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0}
        speed_factor[0] = mapping[label]
    speed_buttons.on_clicked(set_speed)

    plt.show()

    
if __name__ == "__main__":
    plot_test_ukf_bdot(verbose=False, tf=10000, dt=50, t_start_bdot=1000, real_orbit=True)