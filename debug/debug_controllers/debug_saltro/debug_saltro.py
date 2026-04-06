import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

# Ensure local SALTRO build is importable when running from Generalized_ADCS.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

import saltro_py

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW

from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking


def create_saltro_satellite(sat: Satellite) -> object:
    cpp_sat = saltro_py.Satellite()
    cpp_sat.setInertia(sat.J_COM)

    for mtq in sat.mtq_actuators:
        cpp_sat.addMTQ(mtq.axis, mtq.u_max)

    for rw in sat.rw_actuators:
        cpp_sat.addRW(rw.axis, rw.u_max, rw.J, rw.h, rw.h_max)

    return cpp_sat


def _build_common_setup() -> Tuple[Satellite, np.ndarray, Orbital_State, float]:
    mtm_max_torque = 0.2
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 0.001
    rw_J = 1e-5
    rw_h0 = 0.0
    rw_hmax = 0.02
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]

    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(
        mass=4.0,
        J_0=np.diagflat([0.067, 0.071, 0.069]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )

    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([rw_h0, rw_h0, rw_h0])
    x0 = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    t_start = 0.22

    # Keep ADCS-native orbit convention (km, km/s); values match SALTRO debug case.
    R = np.array([7000.0, 0.0, 0.0])
    V = np.array([0.0, 7.5, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=t_start, R=R, V=V)

    return real_sat, x0, os0, t_start


def _configure_like_saltro_debug(planner_settings: PlannerSettings, dt: float) -> None:
    """Apply the same planner settings used by SALTRO debug_3_3_slew90_dt10."""
    planner_settings.init_traj.initcontroller = 2

    p0 = planner_settings.passes[0]
    p0.dt = float(dt)
    p0.ilqr.cost_tol = 1e-5
    p0.ilqr.max_iters = 20

    p0.aug_lag.max_outer_iters = 10
    p0.aug_lag.constraint_tol = 1e-3

    cost = p0.cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = 1

    planner_settings.disturbances.plan_for_aero = 0
    planner_settings.disturbances.plan_for_gg = 0
    planner_settings.disturbances.plan_for_srp = 0
    planner_settings.disturbances.plan_for_prop = 0
    planner_settings.disturbances.plan_for_gendist = 0
    planner_settings.disturbances.plan_for_resdipole = 0

    p0.reg.reg_init = 1e-6
    p0.reg.reg_max = 1e10
    p0.reg.reg_scale = 10.0
    p0.reg.use_dynamics_hess = 0
    p0.reg.use_constraint_hess = 0

    p0.linesearch.max_iters = 24
    p0.linesearch.beta1 = 1e-10
    p0.linesearch.beta2 = 5000.0


def _configure_like_saltro_debug_cpp(cpp_settings, dt: float) -> None:
    """Apply SALTRO debug values directly on C++ planner settings."""
    cpp_settings.init_traj.initcontroller = 2
    cpp_settings.num_passes = 1

    p0 = cpp_settings.passes[0]
    p0.dt = float(dt)
    p0.ilqr.cost_tol = 1e-5
    p0.ilqr.max_iters = 20

    p0.auglag.max_outer_iters = 10
    p0.auglag.constraint_tol = 1e-3

    cost = p0.cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    cpp_settings.disturbances.plan_for_aero = False
    cpp_settings.disturbances.plan_for_gg = False
    cpp_settings.disturbances.plan_for_srp = False
    cpp_settings.disturbances.plan_for_prop = False
    cpp_settings.disturbances.plan_for_gendist = False
    cpp_settings.disturbances.plan_for_resdipole = False

    p0.reg.reg_init = 1e-6
    p0.reg.reg_max = 1e10
    p0.reg.reg_scale = 10.0
    p0.reg.use_dynamics_hess = False
    p0.reg.use_constraint_hess = False

    p0.linesearch.max_iters = 24
    p0.linesearch.beta1 = 1e-10
    p0.linesearch.beta2 = 5000.0


def _run_open_loop_trajopt(
    real_sat: Satellite,
    x0: np.ndarray,
    os0: Orbital_State,
    t_start: float,
    tf: float,
    planner_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planner_settings = PlannerSettings(est_sat=real_sat)
    _configure_like_saltro_debug(planner_settings, dt=planner_dt)
    cpp_settings = planner_settings.to_cpp()
    _configure_like_saltro_debug_cpp(cpp_settings, dt=planner_dt)
    cpp_satellite = create_saltro_satellite(real_sat)

    jtime = np.array(
        [t_start, t_start + tf * TimeConstants.sec2cent],
        dtype=np.float64,
    )

    # Match SALTRO debug_3_3_slew90_dt10 reference definition exactly.
    q_goal = np.array(
        [
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0.0, 0.0],
            [0.0, 0.0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
        ],
        dtype=np.float64,
    )

    boresight = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Convert orbit to SI only at SALTRO boundary.
    r0_m = np.asarray(os0.R, dtype=np.float64) * 1.0e3
    v0_mps = np.asarray(os0.V, dtype=np.float64) * 1.0e3

    ok, X, U, _K = saltro_py.trajOpt(
        cpp_settings,
        cpp_satellite,
        np.asarray(x0, dtype=np.float64),
        r0_m,
        v0_mps,
        jtime,
        q_goal,
        boresight,
    )

    if not ok:
        raise RuntimeError("SALTRO trajOpt failed")

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(U, dtype=np.float64),
        np.asarray(jtime, dtype=np.float64),
        np.asarray(q_goal, dtype=np.float64),
        np.asarray(boresight, dtype=np.float64),
    )


def debug_saltro(
    verbose: bool = False,
    tf: float = 400.0,
    dt: float = 10.0,
    real_orbit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    _ = real_orbit  # Open-loop uses ephemeris propagation here, matching plotted interface needs.

    real_sat, x0, os0, t_start = _build_common_setup()
    X, U, jtime_req, q_goal, _boresight = _run_open_loop_trajopt(
        real_sat=real_sat,
        x0=x0,
        os0=os0,
        t_start=t_start,
        tf=tf,
        planner_dt=dt,
    )

    n_out = X.shape[1]
    if n_out <= jtime_req.shape[0]:
        jtime = jtime_req[:n_out]
    else:
        jtime = np.linspace(
            t_start,
            t_start + tf * TimeConstants.sec2cent,
            n_out,
            dtype=np.float64,
        )

    # Build orbit history on the same timeline used by SALTRO outputs.
    start_time = t_start - 1 * TimeConstants.sec2cent
    end_time = float(jtime[-1]) + 1 * TimeConstants.sec2cent
    orb_os0 = os0.copy()
    orb_os0.J2000 = start_time
    orb = Orbit(os0=orb_os0, end_time=end_time, dt=max(1.0, float(dt)), use_J2=True, fast=False)

    time_hist = (jtime - t_start) / TimeConstants.sec2cent
    state_hist = X.T
    u_hist = U.T
    # SALTRO viewer uses zero-order hold between coarse q_goal knots.
    goal_idx = np.searchsorted(jtime_req[1:], jtime, side="right")
    boresight_hist = q_goal[:, goal_idx].T

    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((n_out, len(real_sat.sensors + real_sat.rw_actuators)))

    for k in range(n_out):
        os_k = orb.get_os(J2000=float(jtime[k]))
        os_hist.append(os_k)
        sensor_hist[k, :] = real_sat.sensor_readings(x=state_hist[k, :], os=os_k)

    if verbose:
        print("SALTRO trajOpt succeeded (open-loop)")
        print(f"N out={n_out}")
        print(f"X shape={X.shape}, U shape={U.shape}")

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist


def plot_mtq_w_rw_align_to_eci(
    verbose: bool = False,
    tf: float = 400.0,
    dt: float = 10.0,
    real_orbit: bool = True,
) -> None:
    time_hist, state_hist, os_hist, _sensor_hist, u_hist, boresight_hist = debug_saltro(
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
    )

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    plot_target_tracking(
        state_hist=state_hist,
        boresight_hist=boresight_hist,
        body_boresight=np.array([0, 0, 1]),
        time=time_hist,
    )
    create_close_all_button_window()


if __name__ == "__main__":
    plot_mtq_w_rw_align_to_eci(verbose=True, tf=400.0, dt=10.0, real_orbit=True)
