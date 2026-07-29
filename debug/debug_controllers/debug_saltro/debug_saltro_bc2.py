import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

# Ensure local SALTRO build is importable when running from Generalized_ADCS.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

import saltro_py

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.state import State

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW

from ADCS.helpers.math_constants import MathConstants


def _boresight_eci(q_b2i: np.ndarray, bore_body_unit: np.ndarray) -> np.ndarray:
    """Transform boresight vector from body frame to inertial frame using quaternion."""
    q0, q1, q2, q3 = q_b2i
    r = np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ],
        dtype=float,
    )
    out = r @ bore_body_unit
    nrm = np.linalg.norm(out)
    return out / max(nrm, 1e-15)


def _vec_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Compute angle between two vectors in degrees."""
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = v / max(np.linalg.norm(v), 1e-15)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def create_saltro_satellite(sat: Satellite, cpp_settings=None) -> object:
    if cpp_settings is not None:
        cpp_sat = saltro_py.Satellite(sat.J_COM, cpp_settings)
    else:
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

    rw_max_torque = 5.7e-6
    rw_J = 0.0023
    rw_h0 = 0.0
    rw_hmax = 0.0036
    rws = [RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)]

    acts = mtqs + rws

    J_0 = np.array(
        [
            [0.03136490806, 5.88304e-05, -0.00671361357],
            [5.88304e-05, 0.03409127827, -0.00012334756],
            [-0.00671361357, -0.00012334756, 0.01004091997],
        ]
    )

    # Match debug_saltro_3+1_reduced.py: no disturbances and no sensors.
    real_sat = Satellite(
        mass=4.0,
        J_0=J_0,
        actuators=acts,
        sensors=[],
        disturbances=[],
        boresight=np.array([0.0, 1.0, 0.0]),
    )

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    t_start = 0.22

    # Match debug_saltro_3+1_reduced.py (km and km/s).
    R = 7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=t_start, R=R, V=V)

    return real_sat, x0, os0, t_start


def _configure_like_bc2_debug(planner_settings: PlannerSettings, dt: float) -> None:
    """Apply debug_bc2.py planner values through Python wrapper settings."""
    planner_settings.init_traj.initcontroller = 2

    p0 = planner_settings.passes[0]
    p0.dt = float(dt)
    p0.ilqr.cost_tol = 1e-5
    p0.ilqr.max_iters = 20

    p0.aug_lag.max_outer_iters = 30
    p0.aug_lag.constraint_tol = 1e-3

    cost = p0.cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_knee_frac = 0.0
    cost.RWh_desat_mult = 0.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 0
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


def _run_open_loop_trajopt(
    real_sat: Satellite,
    x0: np.ndarray,
    os0: Orbital_State,
    t_start: float,
    tf: float,
    planner_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planner_settings = PlannerSettings(est_sat=real_sat)
    _configure_like_bc2_debug(planner_settings, dt=planner_dt)
    cpp_settings = planner_settings.to_cpp()
    cpp_satellite = create_saltro_satellite(real_sat, cpp_settings)

    jtime = np.array(
        [t_start, t_start + tf * TimeConstants.sec2cent],
        dtype=np.float64,
    )

    # Match debug_bc2.py attitude objective.
    vector_goal = np.array(
        [
            [np.nan, np.nan],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    boresight = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Convert orbit to SI at the SALTRO boundary.
    r0_m = np.asarray(os0.R, dtype=np.float64) * 1.0e3
    v0_mps = np.asarray(os0.V, dtype=np.float64) * 1.0e3

    ok, X, U, K = saltro_py.trajOpt(
        cpp_settings,
        cpp_satellite,
        np.asarray(x0, dtype=np.float64),
        r0_m,
        v0_mps,
        jtime,
        vector_goal,
        boresight,
    )

    if not ok:
        raise RuntimeError("SALTRO trajOpt failed")

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(U, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(jtime, dtype=np.float64),
        np.asarray(vector_goal, dtype=np.float64),
        np.asarray(boresight, dtype=np.float64),
    )


def _build_orbit(os0: Orbital_State, t_start: float, t_end: float, dt: float) -> Orbit:
    start_time = t_start - 1 * TimeConstants.sec2cent
    end_time = t_end + 1 * TimeConstants.sec2cent
    orb_os0 = os0.copy()
    orb_os0.J2000 = start_time
    return Orbit(os0=orb_os0, end_time=end_time, dt=max(1.0, float(dt)), zonal_J=2, fast=False)


def _goal_hist_from_knots(jtime_req: np.ndarray, vector_goal: np.ndarray, jtime: np.ndarray) -> np.ndarray:
    goal_idx = np.searchsorted(jtime_req[1:], jtime, side="right")
    return vector_goal[:, goal_idx].T


def debug_saltro(
    verbose: bool = False,
    tf: float = 1000.0,
    dt: float = 5.0,
    real_orbit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ = real_orbit

    real_sat, x0, os0, t_start = _build_common_setup()
    X, U, _K, jtime_req, vector_goal, boresight = _run_open_loop_trajopt(
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

    orb = _build_orbit(os0=os0, t_start=t_start, t_end=float(jtime[-1]), dt=dt)

    time_hist = (jtime - t_start) / TimeConstants.sec2cent
    state_hist = X.T
    u_hist = U.T
    vector_goal_hist = _goal_hist_from_knots(jtime_req=jtime_req, vector_goal=vector_goal, jtime=jtime)

    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((n_out, len(real_sat.sensors + real_sat.rw_actuators)))

    for k in tqdm(range(n_out), desc="Simulating SALTRO open-loop (bc2)"):
        os_k = orb.get_os(J2000=float(jtime[k]))
        os_hist.append(os_k)
        sensor_hist[k, :] = real_sat.sensor_readings(x=State.from_array(state_hist[k, :]), os=os_k)

    if verbose:
        print("SALTRO trajOpt succeeded (open-loop, bc2)")
        print(f"N out={n_out}")
        print(f"X shape={X.shape}, U shape={U.shape}")

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, vector_goal_hist, boresight


def _plot_debug_run(
    tag: str,
    time_hist: np.ndarray,
    state_hist: np.ndarray,
    os_hist: List[Orbital_State],
    u_hist: np.ndarray,
    vector_goal_hist: np.ndarray,
    boresight_body: np.ndarray = None,
) -> None:
    _ = os_hist
    q = state_hist[:, 3:7]
    w = state_hist[:, 0:3]
    h = state_hist[:, 7:]

    n_u = min(len(time_hist), u_hist.shape[0])
    t_u = time_hist[:n_u]
    u_use = u_hist[:n_u, :]

    vec_err = np.zeros(len(q), dtype=np.float64)
    for i in range(len(q)):
        q_i = q[i]
        goal_row = vector_goal_hist[i, :]

        target_vec = np.asarray(goal_row[1:4], dtype=np.float64)
        if np.linalg.norm(target_vec) <= 0.0:
            vec_err[i] = np.nan
            continue

        if boresight_body is None:
            vec_err[i] = np.nan
            continue

        if boresight_body.ndim == 1:
            bore_body = boresight_body
        else:
            goal_idx = np.searchsorted(time_hist, time_hist[i], side="right") if len(time_hist) > 1 else 0
            if goal_idx >= boresight_body.shape[1]:
                goal_idx = boresight_body.shape[1] - 1
            bore_body = boresight_body[:, goal_idx]

        if np.linalg.norm(bore_body) <= 0.0:
            vec_err[i] = np.nan
            continue

        bore_unit = bore_body / np.linalg.norm(bore_body)
        target_unit = target_vec / np.linalg.norm(target_vec)
        bore_inertial = _boresight_eci(q_i, bore_unit)
        vec_err[i] = _vec_angle_deg(bore_inertial, target_unit)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for i in range(4):
        ax.plot(time_hist, q[:, i], linewidth=1.5, label=f"q{i}")
    ax.set_title("Quaternion (BC2 Vector Goal)")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for i, name in enumerate(["wx", "wy", "wz"]):
        ax.plot(time_hist, w[:, i], linewidth=1.5, label=name)
    ax.plot(time_hist, np.linalg.norm(w, axis=1), "k--", linewidth=1.5, label="||w||")
    ax.set_title("Angular Velocity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i in range(u_use.shape[1]):
        ax.plot(t_u, u_use[:, i], linewidth=1.2, label=f"u{i}")
    ax.set_title("Control Inputs")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    if h.shape[1] > 0:
        for i in range(h.shape[1]):
            ax.plot(time_hist, h[:, i], linewidth=1.5, label=f"h_rw{i}")
        ax.legend(fontsize=8)
    ax.set_title("RW Momentum")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(time_hist, vec_err, color="C3", linewidth=1.8)
    ax.set_title("Pointing Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_u, np.linalg.norm(u_use, axis=1), color="C2", linewidth=1.5)
    ax.set_title("Control Magnitude ||u||")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"SALTRO {tag} (BC2 Wrapped) Summary", fontsize=14, fontweight="bold")
    plt.show()
    print(f"Rendered {tag} summary figure (bc2)")


def plot_mtq_w_rw_align_to_eci(
    verbose: bool = False,
    tf: float = 1000.0,
    dt: float = 5.0,
    closed_loop_dt: float = 1.0,
    real_orbit: bool = True,
) -> None:
    _ = closed_loop_dt
    time_hist_ol, state_hist_ol, os_hist_ol, _sensor_hist_ol, u_hist_ol, vector_goal_hist_ol, boresight_ol = debug_saltro(
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
    )
    _plot_debug_run(
        tag="open-loop",
        time_hist=time_hist_ol,
        state_hist=state_hist_ol,
        os_hist=os_hist_ol,
        u_hist=u_hist_ol,
        vector_goal_hist=vector_goal_hist_ol,
        boresight_body=boresight_ol,
    )


if __name__ == "__main__":
    plot_mtq_w_rw_align_to_eci(verbose=True, tf=1000.0, dt=5.0, real_orbit=True)
