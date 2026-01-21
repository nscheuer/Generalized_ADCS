"""
Debug script for Plan and Track LQR controller with BC2 satellite.

This script tests trajectory planning and TVLQR tracking using the ALTRO planner.
Similar to debug_mtq_w_rw_lp_bc2.py but uses trajectory-based control.
"""
import sys
import os as os_module
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple
from tqdm import tqdm
import time as time_module  # For timing instrumentation

sys.path.append(os_module.path.abspath(os_module.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal, Coordinate_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking
from ADCS.helpers.math_helpers import rot_mat
import matplotlib.pyplot as plt


# ============ DEBUG PLOTTING FUNCTIONS ============

# Directory for saving debug plots (AI can read these as images)
DEBUG_PLOT_DIR = os_module.path.join(os_module.path.dirname(__file__), "debug_plots")


def print_trajectory_metrics(time: np.ndarray, state_hist: np.ndarray, u_hist: np.ndarray,
                              goal_vec: np.ndarray = None, title: str = "Trajectory Metrics"):
    """
    Print numerical metrics for trajectory analysis.

    This allows AI to analyze trajectory quality without needing to view plots.
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    N = len(time)

    # Angular velocity metrics
    w = state_hist[:, :3]
    w_mag = np.linalg.norm(w, axis=1)
    w_mag_deg = np.rad2deg(w_mag)
    print(f"\nAngular Velocity:")
    print(f"  Initial |w|: {w_mag_deg[0]:.4f} deg/s")
    print(f"  Final |w|:   {w_mag_deg[-1]:.4f} deg/s")
    print(f"  Max |w|:     {np.max(w_mag_deg):.4f} deg/s (at t={time[np.argmax(w_mag_deg)]:.1f}s)")
    print(f"  Mean |w|:    {np.mean(w_mag_deg):.4f} deg/s")

    # Quaternion / attitude metrics
    errors_deg = None
    if goal_vec is not None:
        errors_deg = []
        for i in range(N):
            q = state_hist[i, 3:7]
            w_s, x_q, y_q, z_q = q
            R = np.array([
                [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w_s), 2*(x_q*z_q + y_q*w_s)],
                [2*(x_q*y_q + z_q*w_s), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w_s)],
                [2*(x_q*z_q - y_q*w_s), 2*(y_q*z_q + x_q*w_s), 1 - 2*(x_q**2 + y_q**2)]
            ])
            body_boresight = np.array([0, 0, 1])
            eci_boresight = R @ body_boresight
            error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_vec), -1, 1))
            errors_deg.append(np.rad2deg(error_rad))
        errors_deg = np.array(errors_deg)
        print(f"\nPointing Error (to goal {goal_vec}):")
        print(f"  Initial error: {errors_deg[0]:.4f} deg")
        print(f"  Final error:   {errors_deg[-1]:.4f} deg")
        print(f"  Max error:     {np.max(errors_deg):.4f} deg (at t={time[np.argmax(errors_deg)]:.1f}s)")
        print(f"  Min error:     {np.min(errors_deg):.4f} deg (at t={time[np.argmin(errors_deg)]:.1f}s)")

    # Control metrics
    n_mtq = min(3, u_hist.shape[1])
    mtq = u_hist[:, :n_mtq]
    print(f"\nMTQ Commands (dipole moment):")
    print(f"  Max |m|:     {np.max(np.abs(mtq)):.6f} A*m^2")
    print(f"  Mean |m|:    {np.mean(np.abs(mtq)):.6f} A*m^2")
    for i in range(n_mtq):
        print(f"  m_{['x','y','z'][i]}: min={np.min(mtq[:,i]):.6f}, max={np.max(mtq[:,i]):.6f}")

    if u_hist.shape[1] > 3:
        rw = u_hist[:, 3:]
        print(f"\nRW Commands (torque):")
        print(f"  Max |tau_rw|:  {np.max(np.abs(rw)):.6e} N*m")
        print(f"  Mean |tau_rw|: {np.mean(np.abs(rw)):.6e} N*m")
        for i in range(rw.shape[1]):
            print(f"  tau_rw{i+1}: min={np.min(rw[:,i]):.6e}, max={np.max(rw[:,i]):.6e}")

    # RW momentum if available
    if state_hist.shape[1] > 7:
        h_rw = state_hist[:, 7:]
        print(f"\nRW Momentum:")
        print(f"  Initial h:   {h_rw[0]} N*m*s")
        print(f"  Final h:     {h_rw[-1]} N*m*s")
        print(f"  Max |h|:     {np.max(np.abs(h_rw)):.6f} N*m*s")

    # Control smoothness (rate of change)
    dt = np.diff(time)
    dt[dt == 0] = 1e-9
    du = np.diff(u_hist[:, :n_mtq], axis=0)
    du_dt = du / dt[:, np.newaxis]
    print(f"\nControl Smoothness (MTQ rate of change):")
    print(f"  Max |dm/dt|: {np.max(np.abs(du_dt)):.6f} A*m^2/s")
    print(f"  Mean |dm/dt|: {np.mean(np.abs(du_dt)):.6f} A*m^2/s")

    # Check for oscillations (sign changes)
    sign_changes = np.sum(np.diff(np.sign(mtq), axis=0) != 0, axis=0)
    print(f"  Sign changes per axis: x={sign_changes[0]}, y={sign_changes[1]}, z={sign_changes[2]}")

    # Zero-order hold detection
    zero_changes = np.sum(np.abs(du) < 1e-10, axis=0)
    print(f"  Zero-change steps per axis: x={zero_changes[0]}, y={zero_changes[1]}, z={zero_changes[2]} (out of {N-1})")

    print(f"{'='*60}\n")

    return {
        'final_w_deg': w_mag_deg[-1],
        'final_error_deg': errors_deg[-1] if errors_deg is not None else None,
        'max_mtq': np.max(np.abs(mtq)),
        'sign_changes': sign_changes,
        'zero_changes': zero_changes,
    }


def save_debug_plots(figs: list, prefix: str = "debug"):
    """Save figures to PNG files for AI analysis."""
    os_module.makedirs(DEBUG_PLOT_DIR, exist_ok=True)
    saved_paths = []
    for i, fig in enumerate(figs):
        if fig is not None:
            path = os_module.path.join(DEBUG_PLOT_DIR, f"{prefix}_{i}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_paths.append(path)
            print(f"  Saved: {path}")
    return saved_paths


def plot_control_scaled(time: np.ndarray, u_hist: np.ndarray, title: str = "Control Commands", save_path: str = None):
    """Plot controls with separate subplots for MTQ and RW with proper scaling."""
    u = np.asarray(u_hist)
    N, M = u.shape
    n_mtq = min(3, M)
    n_rw = M - n_mtq

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # MTQ subplot
    ax = axes[0]
    colors = ['b', 'g', 'r']
    for i in range(n_mtq):
        ax.plot(time[:N], u[:, i], color=colors[i], label=f'MTQ_{["x","y","z"][i]}', alpha=0.8)
    ax.set_ylabel('MTQ Dipole [A$\\cdot$m$^2$]')
    ax.set_title(f'{title} - Magnetorquers')
    ax.legend(loc='upper right')
    ax.grid(True)

    # RW subplot
    ax = axes[1]
    if n_rw > 0:
        for i in range(n_rw):
            ax.plot(time[:N], u[:, n_mtq + i], label=f'RW_{i+1}', alpha=0.8)
        ax.set_ylabel('RW Torque [N$\\cdot$m]')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No RW', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time [s]')
    ax.set_title(f'{title} - Reaction Wheels')
    ax.grid(True)

    fig.tight_layout()

    if save_path:
        os_module.makedirs(os_module.path.dirname(save_path) if os_module.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_trajectory_debug(time: np.ndarray, state_hist: np.ndarray, u_hist: np.ndarray,
                          B_eci: np.ndarray, title_prefix: str = "Trajectory", save_dir: str = None):
    """Comprehensive debug plots: B-field, MTQ torque, control rates."""
    N = min(len(time), len(state_hist), len(u_hist))
    time = time[:N]
    state_hist = state_hist[:N]
    u_hist = u_hist[:N]

    B = np.asarray(B_eci)
    if B.shape[0] == 3 and len(B.shape) == 2 and B.shape[1] != 3:
        B = B.T
    B = B[:N]

    # Transform B-field to body frame
    B_body = np.zeros((N, 3))
    for i in range(N):
        q = state_hist[i, 3:7]
        R = rot_mat(q)
        B_body[i] = R.T @ B[i]

    # Figure 1: B-field and MTQ torque
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes1[0, 0]
    ax.plot(time, B[:, 0] * 1e6, label='$B_x$', alpha=0.8)
    ax.plot(time, B[:, 1] * 1e6, label='$B_y$', alpha=0.8)
    ax.plot(time, B[:, 2] * 1e6, label='$B_z$', alpha=0.8)
    ax.set_ylabel('B-field [$\\mu$T]')
    ax.set_title(f'{title_prefix}: B-field (ECI)')
    ax.legend()
    ax.grid(True)

    ax = axes1[0, 1]
    ax.plot(time, B_body[:, 0] * 1e6, label='$B_x$', alpha=0.8)
    ax.plot(time, B_body[:, 1] * 1e6, label='$B_y$', alpha=0.8)
    ax.plot(time, B_body[:, 2] * 1e6, label='$B_z$', alpha=0.8)
    ax.set_ylabel('B-field [$\\mu$T]')
    ax.set_title(f'{title_prefix}: B-field (Body)')
    ax.legend()
    ax.grid(True)

    ax = axes1[1, 0]
    ax.plot(time, u_hist[:, 0], label='$m_x$', alpha=0.8)
    ax.plot(time, u_hist[:, 1], label='$m_y$', alpha=0.8)
    ax.plot(time, u_hist[:, 2], label='$m_z$', alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MTQ dipole [A$\\cdot$m$^2$]')
    ax.set_title(f'{title_prefix}: MTQ Commands')
    ax.legend()
    ax.grid(True)

    m = u_hist[:, :3]
    tau = np.cross(m, B_body)
    ax = axes1[1, 1]
    ax.plot(time, tau[:, 0] * 1e6, label='$\\tau_x$', alpha=0.8)
    ax.plot(time, tau[:, 1] * 1e6, label='$\\tau_y$', alpha=0.8)
    ax.plot(time, tau[:, 2] * 1e6, label='$\\tau_z$', alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque [$\\mu$N$\\cdot$m]')
    ax.set_title(f'{title_prefix}: MTQ Torque ($\\tau = m \\times B$)')
    ax.legend()
    ax.grid(True)

    fig1.tight_layout()

    # Figure 2: Angular velocity and RW
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))

    w = state_hist[:, :3]
    ax = axes2[0, 0]
    ax.plot(time, np.rad2deg(w[:, 0]), label='$\\omega_x$', alpha=0.8)
    ax.plot(time, np.rad2deg(w[:, 1]), label='$\\omega_y$', alpha=0.8)
    ax.plot(time, np.rad2deg(w[:, 2]), label='$\\omega_z$', alpha=0.8)
    ax.set_ylabel('$\\omega$ [deg/s]')
    ax.set_title(f'{title_prefix}: Angular Velocity')
    ax.legend()
    ax.grid(True)

    ax = axes2[0, 1]
    ax.plot(time, np.rad2deg(np.linalg.norm(w, axis=1)), 'k-')
    ax.set_ylabel('$|\\omega|$ [deg/s]')
    ax.set_title(f'{title_prefix}: Angular Velocity Magnitude')
    ax.grid(True)

    # Control rates
    dt = np.diff(time)
    dt[dt == 0] = 1e-9
    du = np.diff(u_hist[:, :3], axis=0)
    du_dt = du / dt[:, np.newaxis]
    time_mid = (time[:-1] + time[1:]) / 2

    ax = axes2[1, 0]
    ax.plot(time_mid, du_dt[:, 0], label='$dm_x/dt$', alpha=0.8)
    ax.plot(time_mid, du_dt[:, 1], label='$dm_y/dt$', alpha=0.8)
    ax.plot(time_mid, du_dt[:, 2], label='$dm_z/dt$', alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MTQ rate [A$\\cdot$m$^2$/s]')
    ax.set_title(f'{title_prefix}: MTQ Control Rates')
    ax.legend()
    ax.grid(True)

    ax = axes2[1, 1]
    if u_hist.shape[1] > 3:
        ax.plot(time, u_hist[:, 3], 'b-', label='RW torque', linewidth=1.5)
        ax.set_ylabel('RW torque [N$\\cdot$m]', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        if state_hist.shape[1] > 7:
            ax2 = ax.twinx()
            ax2.plot(time, state_hist[:, 7], 'r--', label='RW momentum', linewidth=1.5)
            ax2.set_ylabel('RW momentum [N$\\cdot$m$\\cdot$s]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_title(f'{title_prefix}: Reaction Wheel')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No RW data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title_prefix}: Reaction Wheel')

    fig2.tight_layout()

    if save_dir:
        os_module.makedirs(save_dir, exist_ok=True)
        path1 = os_module.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_bfield_torque.png")
        path2 = os_module.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_angvel_rw.png")
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path1}")
        print(f"  Saved: {path2}")

    return fig1, fig2


def patch_planner_with_timing_and_env(controller):
    """Monkey-patch to add timing and capture environment data for debug plots."""
    controller._debug_env_data = {}

    def timed_calculate_trajectory_common(t_start, duration, x_0, os_0, goals, verbose=False,
                                          vecsPy_precomputed=None, N_precomputed=None, t_end_precomputed=None):
        from ADCS.orbits.universal_constants import TimeConstants

        if verbose:
            print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")

        controller.planner.setVerbosity(verbose)
        dt_seconds = controller.planner_settings.dt_tvlqr

        # Use precomputed values if provided, otherwise compute
        if vecsPy_precomputed is not None:
            vecsPy = vecsPy_precomputed
            N = N_precomputed
            t_end = t_end_precomputed
            print(f"  [TIMING] Using precomputed environment data")
        else:
            N = int(np.ceil(duration / dt_seconds)) + 1
            t_end = t_start + (duration * TimeConstants.sec2cent)
            t_env_start = time_module.perf_counter()
            vecsPy = controller._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)
            t_env_end = time_module.perf_counter()
            print(f"  [TIMING] _propagate_environment: {t_env_end - t_env_start:.2f}s")

        # Store for debug plotting: vecsPy = (t, R, V, B, S, A, E, p, rho)
        controller._debug_env_data = {
            'times': vecsPy[0], 'B_eci': vecsPy[3], 'S_eci': vecsPy[4],
            'goal_eci': vecsPy[6], 'rho': vecsPy[8],
        }

        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        bdotOn = controller.planner_settings.bdot_on

        t_altro_start = time_module.perf_counter()
        (_, _, _, lqr_opt, _) = controller.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn))
        t_altro_end = time_module.perf_counter()
        print(f"  [TIMING] trajOpt (ALTRO): {t_altro_end - t_altro_start:.2f}s")

        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt

        from ADCS.controller.helpers import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python
        Uset = reorder_controls_cpp_to_python(Uset_cpp, controller.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, controller.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)

    controller._calculate_trajectory_common = timed_calculate_trajectory_common
    return controller


def patch_planner_with_timing(controller):
    """Monkey-patch the planner to add timing instrumentation for ALTRO vs orbit propagation."""
    original_calculate_common = controller._calculate_trajectory_common

    def timed_calculate_trajectory_common(t_start, duration, x_0, os_0, goals, verbose=False,
                                          vecsPy_precomputed=None, N_precomputed=None, t_end_precomputed=None):
        from ADCS.orbits.universal_constants import TimeConstants
        import numpy as np

        if verbose:
            print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")

        controller.planner.setVerbosity(verbose)
        dt_seconds = controller.planner_settings.dt_tvlqr

        # Use precomputed values if provided, otherwise compute
        if vecsPy_precomputed is not None:
            vecsPy = vecsPy_precomputed
            N = N_precomputed
            t_end = t_end_precomputed
            print(f"  [TIMING] Using precomputed environment data")
        else:
            N = int(np.ceil(duration / dt_seconds)) + 1
            t_end = t_start + (duration * TimeConstants.sec2cent)
            t_env_start = time_module.perf_counter()
            vecsPy = controller._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)
            t_env_end = time_module.perf_counter()
            print(f"  [TIMING] _propagate_environment (planner orbit): {t_env_end - t_env_start:.2f}s")

        # SANITIZE x_0
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')
        bdotOn = controller.planner_settings.bdot_on

        # Time the actual ALTRO trajOpt call
        t_altro_start = time_module.perf_counter()
        (_, _, _, lqr_opt, _) = controller.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn))
        t_altro_end = time_module.perf_counter()
        print(f"  [TIMING] trajOpt (ALTRO C++): {t_altro_end - t_altro_start:.2f}s")

        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt

        # Reorder controls and gains
        from ADCS.controller.helpers import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python
        Uset = reorder_controls_cpp_to_python(Uset_cpp, controller.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, controller.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)

    controller._calculate_trajectory_common = timed_calculate_trajectory_common
    return controller


def test_plan_and_track_lqr(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 1,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, Trajectory]:
    """
    Test the Plan and Track LQR controller with BC2 satellite.

    Args:
        verbose: Print debug information
        tf: Final time in seconds
        dt: Simulation timestep in seconds
        dt_planning: Trajectory planner timestep in seconds
        real_orbit: Use real orbit propagation (True) or simplified orbit (False)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, trajectory)
    """
    np.random.seed(seed)
    t0 = 0
    N = int((tf - t0) / dt)

    # Create BC2 satellite
    rw_h0 = 0.0
    real_sat = create_beavercube2_cubesat(estimated=False)
    real_sat.rw_actuators[0].h = rw_h0

    # Initial conditions (same style as quick_planner_tests)
    w0 = np.random.randn(3) * 0.01  # ~0.5 deg/s typical
    q0 = normalize(np.random.randn(4))
    h0 = np.array([rw_h0])
    x = np.concatenate([w0, q0, h0])

    print(f"Initial angular velocity: {np.rad2deg(np.linalg.norm(w0)):.2f} deg/s")
    print(f"Initial quaternion: {q0}")

    # Create orbit
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + (tf - t0) * TimeConstants.sec2cent
    R = 7000e3 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])  # meters
    V = np.array([8000, 0, 0])  # m/s

    if real_orbit:
        print("Creating real orbit (this may take a moment)...")
        t_orbit_start = time_module.perf_counter()
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=True)
        t_orbit_end = time_module.perf_counter()
        print(f"  [TIMING] Initial orbit creation: {t_orbit_end - t_orbit_start:.2f}s")
    else:
        os0 = Orbital_State(
            ephem=ephem,
            J2000=0.22 - 1 * TimeConstants.sec2cent,
            R=R,
            V=V,
            B=np.array([0, 0.1, 0]),
            S=np.array([1e5 + 1, 0, 0]),
            rho=5e-12,
        )
        dur = int((tf - t0) / dt) + 10
        orbs = [os0] * (dur + 10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
        orb = Orbit(orbs)

    # Setup planner with tuned settings (see ALTRO_TUNING_NOTES.md)
    print("Setting up trajectory planner...")
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=1,  # Skip bdot initial guess (faster, more reliable)
        dt_tp=dt_planning,
        dt_tvlqr=dt,
    )
    planner_settings.verbosity = verbose

    # Tuned cost weights: low running + high terminal = fast convergence
    planner_settings.cost_main.use_full_cost_hessian = True
    planner_settings.pass1.regularization.use_dynamics_hess = 1
    planner_settings.cost_second.use_full_cost_hessian = True
    planner_settings.pass2.regularization.use_dynamics_hess = 0
    planner_settings.init_traj.bdot_gain = 500
    planner_settings.cost_main.angle =     10000
    planner_settings.cost_main.angle_N =   1000000
    planner_settings.cost_second.angle =   10000
    planner_settings.cost_second.angle_N = 1000000
    planner_settings.cost_main.ang_vel = 1e2
    # planner_settings.cost_main.ang_vel_N = 1e4
    planner_settings.cost_second.ang_vel = 1e2
    # planner_settings.cost_second.ang_vel_N = 1e4
    planner_settings.cost_second.ang_cost_func_type = 2
    planner_settings.cost_second.ang_cost_func_type = 2 
    planner_settings.pass1.aug_lag.penalty_init = 100
    planner_settings.pass2.aug_lag.penalty_init = 1000
    planner_settings.pass1.convergence.max_outer_iter = 15 
    planner_settings.pass1.convergence.max_inner_iter = 50
    planner_settings.pass2.convergence.max_outer_iter = 20
    planner_settings.pass2.convergence.max_inner_iter = 30

    
    planner_settings.rw_control_weight = 1e2
    planner_settings.mtq_control_weight = 1e2
    # planner_settings.wmax = 10*np.pi/180.0


    # ============ COST WEIGHTS TUNED FOR FASTER CONVERGENCE ============
    # Key insight: Lower running costs + high terminal costs = faster convergence
    # The optimizer can take "shortcuts" during trajectory but must hit goal
    # planner_settings.cost_main.ang_vel = 1e1   # Lower running cost (was 1e4)
    # planner_settings.cost_second.ang_vel = 1e1
    # planner_settings.cost_tvlqr.ang_vel = 1e1
    # planner_settings.cost_main.ang_vel_N = 1e2     # Higher terminal (was 1e6)
    # planner_settings.cost_second.ang_vel_N = 1e2
    # planner_settings.cost_tvlqr.ang_vel_N = 1e2
    # planner_settings.cost_main.angle = 100         # Lower running cost (was 1e8)
    # planner_settings.cost_second.angle = 1e2
    # planner_settings.cost_tvlqr.angle = 1e2
    # planner_settings.cost_main.angle_N = 1e4      # Keep high terminal
    # planner_settings.cost_second.angle_N = 1e4
    # planner_settings.cost_tvlqr.angle_N = 1e4
    # =====================================================================

    # Pass 1: use_raw_control_cost=True penalizes |u|, allowing free control exploration
    # Pass 2: use_raw_control_cost=False penalizes |u-u_prev|, smoothing the trajectory
    # If Pass 1 is stuck (dLA=0), the delta-cost is trapping it at initial trajectory
    planner_settings.plan_for_aero = True
    planner_settings.plan_for_srp = True
    planner_settings.plan_for_gg = True
    # planner_settings.cost_tvlqr.control_mult = 1e0
    # planner_settings.cost_second.control_mult = 1e0

    # ============ FAST MC SETTINGS (bdot_on=2 + Gauss-Newton) ============
    # Optimized via parameter sweep: ~4s ALTRO time with good trajectory quality
    # Key: Focus iterations on pass1, minimal pass2, use Gauss-Newton (no full Hessian)
    # "Focus pass1 no Hess": 3.90s ALTRO, 0.92°/s vel, 0.48° error
    # planner_settings.pass1.convergence.max_outer_iter = 20   # Focus on pass1
    # planner_settings.pass1.convergence.max_inner_iter = 100
    # planner_settings.pass2.convergence.max_outer_iter = 5  # Minimal pass2
    # planner_settings.pass2.convergence.max_inner_iter = 30

    # # Relaxed tolerances for speed
    # planner_settings.pass1.convergence.grad_tol = 0.01
    # # # planner_settings.pass1.convergence.ilqr_cost_tol = 1
    # # # planner_settings.pass1.convergence.c_max = 0.1
    # planner_settings.pass2.convergence.grad_tol = 0.001
    # # planner_settings.pass2.convergence.ilqr_cost_tol = 1
    # planner_settings.pass2.convergence.c_max = 0.01
    

    # Use Gauss-Newton approximation (faster per iteration, sufficient for MC)
    # planner_settings.cost_main.use_full_cost_hessian = False
    # planner_settings.cost_second.use_full_cost_hessian = True
    # planner_settings.cost_tvlqr.use_full_cost_hessian = False
    # planner_settings.pass1.regularization.use_dynamics_hess = 0
    # planner_settings.pass2.regularization.use_dynamics_hess = 1
    # ==========================================================================



    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    # Goal setup
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({0.22: goal})

    # Calculate trajectory
    print("Calculating trajectory...")
    os0_for_traj = orb.get_os(0.22)
    try:
        traj: Trajectory = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf - t0,
            x_0=x,
            os_0=os0_for_traj,
            goals=goals,
            verbose=verbose,
        )
        controller.set_active_trajectory(traj)
        traj_duration_centuries = traj.end_time - traj.start_time
        traj_duration_seconds = traj_duration_centuries / TimeConstants.sec2cent
        print(f"Trajectory calculated successfully!")
        print(f"  Start: {traj.start_time:.6f}, End: {traj.end_time:.6f} (J2000 centuries)")
        print(f"  Duration: {traj_duration_seconds:.1f}s")
        print(f"  N steps: {traj.n_steps}, Gains shape: {traj.gains.shape}")
    except Exception as e:
        print(f"Trajectory calculation failed: {e}")
        raise
    time_hist_traj = (traj.times-start_time)*TimeConstants.cent2sec
    state_hist_traj = traj.states.T
    u_hist_traj = traj.controls.T

    # Print trajectory metrics for AI analysis (doesn't require viewing plots)
    print_trajectory_metrics(time_hist_traj, state_hist_traj, u_hist_traj,
                             goal_vec=goal_vec, title="PLANNED TRAJECTORY METRICS")

    # Standard plots (saved to files for AI to read as images)
    plot_state_comparison(time=time_hist_traj, state_hist=state_hist_traj)
    ctrl_fig = plot_control_scaled(time=time_hist_traj, u_hist=u_hist_traj, title="Planned Trajectory Controls",
                                   save_path=os_module.path.join(DEBUG_PLOT_DIR, "planned_controls.png"))

    boresight_traj_hist = np.vstack([goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0] for J2000 in traj.times])
    plot_target_tracking(state_hist=state_hist_traj, boresight_hist=boresight_traj_hist, body_boresight=np.array([0, 0, 1]))
    plot_rw_momentum(time=time_hist_traj, state_hist=state_hist_traj)

    # Debug plots with B-field and torque (saved to files)
    if hasattr(controller, '_debug_env_data') and 'B_eci' in controller._debug_env_data:
        B_eci = controller._debug_env_data['B_eci']
        plot_trajectory_debug(time_hist_traj, state_hist_traj, u_hist_traj, B_eci,
                              title_prefix="Planned_Traj", save_dir=DEBUG_PLOT_DIR)

    create_close_all_button_window()

    # Initialize history arrays
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(real_sat.actuators)))
    boresight_hist = np.nan * np.zeros((N, 3))

    # Simulation loop
    t = t0
    ind = 0
    steps = int((tf - t0) / dt)

    print(f"Running simulation for {tf}s with dt={dt}s...")
    for step in tqdm(range(steps), desc="Simulating Plan & Track LQR"):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)

        # Get control from TVLQR tracking
        try:
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)
        except RuntimeError as e:
            print(f"Controller error at t={t}: {e}")
            break

        # if verbose:
        #     print(f"t={t:.1f}s, u={u}")

        # Store history
        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        eci_goal, w_goal = goal.to_ref(os0=os)
        boresight_hist[ind, :] = eci_goal

        # Propagate dynamics
        ind += 1
        t += dt
        prev_os = os.copy()
        os_next = orb.get_os(0.22 + (t - t0) * TimeConstants.sec2cent)

        out = solve_ivp(
            fun=real_sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, prev_os, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj


def plot_plan_and_track_lqr(
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 1,
    dt_planning: float = 1,
    real_orbit: bool = True,
    seed: int = 37,
) -> None:
    """
    Run and plot the Plan and Track LQR controller test.
    """
    results = test_plan_and_track_lqr(
        verbose=verbose,
        tf=tf,
        dt=dt,
        dt_planning=dt_planning,
        real_orbit=real_orbit,
        seed=seed,
    )
    time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist, traj = results

    # Trim NaN values
    valid_idx = ~np.isnan(time_hist)
    time_hist = time_hist[valid_idx]
    state_hist = state_hist[valid_idx]
    u_hist = u_hist[valid_idx]
    boresight_hist = boresight_hist[valid_idx]

    print(f"\n--- Simulation Complete ---")
    print(f"Final angular velocity: {np.rad2deg(np.linalg.norm(state_hist[-1, :3])):.4f} deg/s")
    print(f"Final quaternion: {state_hist[-1, 3:7]}")

    # Calculate final tracking error
    q_final = state_hist[-1, 3:7]
    # Rotation matrix from body to inertial
    w, x_q, y_q, z_q = q_final
    R = np.array([
        [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w), 2*(x_q*z_q + y_q*w)],
        [2*(x_q*y_q + z_q*w), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w)],
        [2*(x_q*z_q - y_q*w), 2*(y_q*z_q + x_q*w), 1 - 2*(x_q**2 + y_q**2)]
    ])
    body_boresight = np.array([0, 0, 1])
    eci_boresight = R @ body_boresight
    goal_eci = boresight_hist[-1]
    error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_eci), -1, 1))
    print(f"Final tracking error: {np.rad2deg(error_rad):.4f} deg")

    # Plot results
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_target_tracking(state_hist=state_hist, boresight_hist=boresight_hist, body_boresight=np.array([0, 0, 1]))

    create_close_all_button_window()


if __name__ == "__main__":
    plot_plan_and_track_lqr(
        verbose=3,  # Quiet mode
        tf=250,  # 60s trajectory for testing
        dt=1,
        dt_planning=30,  # Tuned setting from quick_planner_tests
        real_orbit=True,  # Use fast orbit (use_J2=False) for faster testing
        seed=42,  # Same seed as quick_planner_tests
    )
