import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import pytest
from functools import lru_cache

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import (
    plot_control,
    plot_rw_momentum,
    plot_target_tracking,
)
from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista


# ----------------------------
# Scenario configs
# ----------------------------
POINTING_CFG: Dict[str, float] = dict(
    p_gain=0.00005,
    d_gain=0.001,
    c_gain=0.0,
    h_target=np.array([0.004]),  # One per RW (test uses 1 RW)
)

DESAT_CFG: Dict[str, float] = dict(
    p_gain=0.00005,
    d_gain=0.00005,
    c_gain=0.02,
    h_target=np.array([0.002]),  # One per RW (test uses 1 RW)
)

CTRL_EFFORT_TOL = 0.01  # magnitude threshold at end (except ground tracking)


# ----------------------------
# Shared setup + utilities
# ----------------------------
def _make_satellite() -> Tuple[Satellite, np.ndarray, List]:
    mtq_max_moment = 0.4
    mtqs = [MTQ(axis=j, max_moment=mtq_max_moment) for j in MathConstants.unitvecs]

    rw_max_torque = 7 * 0.001
    rw_J = 0.001
    rw_h0 = 5 * 0.001
    rw_hmax = 16.2 * 0.001

    rws = [
        RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
        for j in MathConstants.unitvecs
    ]
    # Keep 1 RW: remove two
    rws.pop()
    rws.pop()

    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    sat = Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )

    # Initial state
    w0 = random_n_unit_vec(3) * np.random.uniform(1, 2) * np.pi / 180.0
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    h0 = np.array([rw_h0])
    x0 = np.concatenate([w0, q0, h0])

    return sat, x0, acts


def _make_controller(est_sat: Satellite, goal: Goal) -> MTQ_w_RW_LP:
    cfg = DESAT_CFG if isinstance(goal, No_Goal) else POINTING_CFG
    return MTQ_w_RW_LP(
        est_sat=est_sat,
        p_gain=cfg["p_gain"],
        d_gain=cfg["d_gain"],
        c_gain=cfg["c_gain"],
        h_target=cfg["h_target"],
    )


@lru_cache(maxsize=16)
def _get_real_orbit_cached(tf: float, dt: float) -> Orbit:
    """
    Cached real orbit propagation so that pytest runs don't propagate 4x.
    Cache key is (tf, dt). If you vary those, you'll get a new propagation.
    """
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent

    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])

    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)


def _make_fake_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])

    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - 1 * TimeConstants.sec2cent,
        R=R,
        V=V,
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )

    dur = int(tf / dt) + 10
    orbs = [os0] * (dur + 10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
    return Orbit(orbs)


def _quat_to_dcm_body_to_eci(q: np.ndarray) -> np.ndarray:
    """
    Assumes scalar-first quaternion q = [q0, q1, q2, q3] and returns DCM for body->ECI.
    """
    q = normalize(q)
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)],
    ])


def _final_pointing_error_deg(
    state_hist: np.ndarray,
    boresight_goal_hist: np.ndarray,
    body_boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> float:
    valid = np.where(~np.isnan(state_hist[:, 0]))[0]
    if len(valid) == 0:
        return float("nan")
    k = valid[-1]

    q = state_hist[k, 3:7]
    goal_eci = boresight_goal_hist[k, :]

    R_b2i = _quat_to_dcm_body_to_eci(q)
    boresight_eci = R_b2i @ body_boresight

    boresight_eci = boresight_eci / (np.linalg.norm(boresight_eci) + 1e-16)
    goal_eci = goal_eci / (np.linalg.norm(goal_eci) + 1e-16)

    c = float(np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _final_control_effort(u_hist: np.ndarray) -> float:
    valid = np.where(~np.isnan(u_hist[:, 0]))[0]
    if len(valid) == 0:
        return float("nan")
    k = valid[-1]
    return float(np.linalg.norm(u_hist[k, :]))


def simulate_MTQ_w_RW_LP(
    goal: Goal,
    verbose: bool = False,
    tf: float = 1000,
    dt: float = 10,
    real_orbit: bool = True,
    orbit: Optional[Orbit] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1)
    t0 = 0.0
    steps = int((tf - t0) / dt)
    N = steps

    sat, x, acts = _make_satellite()
    controller = _make_controller(est_sat=sat, goal=goal)

    if orbit is None:
        orbit = _get_real_orbit_cached(tf=tf, dt=dt) if real_orbit else _make_fake_orbit(tf=tf, dt=dt)

    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist: np.ndarray = np.nan * np.zeros((N, len(sat.sensors + sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(acts)))
    boresight_hist = np.nan * np.zeros((N, 3))

    t = t0
    ind = 0

    for _ in tqdm(range(steps), desc="Simulating MTQ_w_RW_LP"):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os_now = orbit.get_os(J2000=J2000)

        sens = sat.sensor_readings(x=x, os=os_now)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os_now)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        eci_goal, _w_goal = goal.to_ref(os0=os_now)
        boresight_hist[ind, :] = eci_goal

        # propagate dynamics
        t_next = t + dt
        os_prev = os_now.copy()
        os_next = orbit.get_os(J2000=0.22 + t_next * TimeConstants.sec2cent)

        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os_prev, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        t = t_next
        ind += 1

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist


# ----------------------------
# Pytest: convergence + final control effort checks
# ----------------------------
@pytest.mark.slow
def test_mtq_w_rw_ps_1rw_hold_converges(tf: float = 500, dt: float = 2) -> None:
    orbit = _get_real_orbit_cached(tf=tf, dt=dt)
    goal = ECI_Goal(np.array([0.0, 0.0, 1.0]))
    _, state_hist, _, _, u_hist, boresight_hist = simulate_MTQ_w_RW_LP(goal=goal, tf=tf, dt=dt, real_orbit=True, orbit=orbit)
    err_deg = _final_pointing_error_deg(state_hist, boresight_hist)
    assert err_deg <= 0.1, f"Final pointing error too large: {err_deg:.3f} deg"
    u_end = _final_control_effort(u_hist)
    assert u_end <= CTRL_EFFORT_TOL, f"Final control effort too large: {u_end:.4f} (tol={CTRL_EFFORT_TOL})"


@pytest.mark.slow
def test_mtq_w_rw_ps_1rw_easy_turn_converges(tf: float = 500, dt: float = 2) -> None:
    orbit = _get_real_orbit_cached(tf=tf, dt=dt)
    goal = ECI_Goal(np.array([0.0, 1.0, 0.0]))
    _, state_hist, _, _, u_hist, boresight_hist = simulate_MTQ_w_RW_LP(goal=goal, tf=tf, dt=dt, real_orbit=True, orbit=orbit)
    err_deg = _final_pointing_error_deg(state_hist, boresight_hist)
    assert err_deg <= 0.5, f"Final pointing error too large: {err_deg:.3f} deg"
    u_end = _final_control_effort(u_hist)
    assert u_end <= CTRL_EFFORT_TOL, f"Final control effort too large: {u_end:.4f} (tol={CTRL_EFFORT_TOL})"


@pytest.mark.slow
def test_mtq_w_rw_ps_1rw_hard_turn_converges(tf: float = 500, dt: float = 2) -> None:
    orbit = _get_real_orbit_cached(tf=tf, dt=dt)
    goal = ECI_Goal(np.array([1.0, 1.0, 1.0]))
    _, state_hist, _, _, u_hist, boresight_hist = simulate_MTQ_w_RW_LP(goal=goal, tf=tf, dt=dt, real_orbit=True, orbit=orbit)
    err_deg = _final_pointing_error_deg(state_hist, boresight_hist)
    assert err_deg <= 0.5, f"Final pointing error too large: {err_deg:.3f} deg"
    u_end = _final_control_effort(u_hist)
    assert u_end <= CTRL_EFFORT_TOL, f"Final control effort too large: {u_end:.4f} (tol={CTRL_EFFORT_TOL})"


@pytest.mark.slow
def test_mtq_w_rw_ps_1rw_ground_tracking_converges(tf: float = 500, dt: float = 2) -> None:
    orbit = _get_real_orbit_cached(tf=tf, dt=dt)
    goal = Coordinate_Goal(lat=9, lon=-70, alt=0)
    _, state_hist, _, _, _u_hist, boresight_hist = simulate_MTQ_w_RW_LP(goal=goal, tf=tf, dt=dt, real_orbit=True, orbit=orbit)
    err_deg = _final_pointing_error_deg(state_hist, boresight_hist)
    assert err_deg <= 5.0, f"Final pointing error too large: {err_deg:.3f} deg"
    # NOTE: no final control-effort check for ground tracking (tracking is time-varying)


@pytest.mark.slow
def test_mtq_w_rw_ps_1rw_desat_converges(tf: float = 2500, dt: float = 5) -> None:
    orbit = _get_real_orbit_cached(tf=tf, dt=dt)
    goal = No_Goal()
    _, state_hist, _, _, u_hist, _boresight_hist = simulate_MTQ_w_RW_LP(goal=goal, tf=tf, dt=dt, real_orbit=True, orbit=orbit)
    u_end = _final_control_effort(u_hist)
    assert u_end <= CTRL_EFFORT_TOL, f"Final control effort too large: {u_end:.4f} (tol={CTRL_EFFORT_TOL})"
    valid = np.where(~np.isnan(state_hist[:, 0]))[0]
    assert len(valid) > 0, "No valid state history found"
    k = valid[-1]
    omega_final = state_hist[k, 0:3]
    omega_norm = float(np.linalg.norm(omega_final))
    assert omega_norm <= 1e-4, (
        f"Final angular rate too large: ||ω|| = {omega_norm:.6e} rad/s (tol=1e-4)"
    )


# ----------------------------
# Plot/debug entrypoints (still works with __main__)
# ----------------------------
def plot_scenario(
    scenario: str,
    verbose: bool = False,
    tf: float = 500,
    dt: float = 2,
    real_orbit: bool = True,
) -> None:
    scenario = scenario.lower().strip()

    if scenario in ("hold", "eci_hold", "align_z"):
        goal = ECI_Goal(np.array([0.0, 0.0, 1.0]))
    elif scenario in ("easy", "easy_turn", "turn_y"):
        goal = ECI_Goal(np.array([0.0, 1.0, 0.0]))
    elif scenario in ("hard", "hard_turn", "turn_diag"):
        goal = ECI_Goal(np.array([1.0, 1.0, 1.0]))
    elif scenario in ("ground", "ground_tracking", "coord"):
        goal = Coordinate_Goal(lat=40, lon=-40, alt=0)
    elif scenario in ("desat", "no_goal"):
        goal = No_Goal()
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Try: hold, easy_turn, hard_turn, ground_tracking, desat")

    orbit = _get_real_orbit_cached(tf=tf, dt=dt) if real_orbit else None
    time_hist, state_hist, os_hist, _sensor_hist, u_hist, boresight_hist = simulate_MTQ_w_RW_LP(
        goal=goal,
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
        orbit=orbit,
    )

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)

    coord_goal = goal if isinstance(goal, Coordinate_Goal) else None
    animate_orbit_pyvista(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        boresight_goal_hist=boresight_hist,
        coord_goal=coord_goal,
    )
    plot_target_tracking(
        state_hist=state_hist,
        boresight_hist=boresight_hist,
        body_boresight=np.array([0.0, 0.0, 1.0]),
    )
    create_close_all_button_window()


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "desat"

    # Default sim horizon depends on scenario
    if scenario.lower().strip() in ("desat", "no_goal"):
        tf_default, dt_default = 2500, 5
    else:
        tf_default, dt_default = 500, 2

    plot_scenario(
        scenario=scenario,
        verbose=False,
        tf=tf_default,
        dt=dt_default,
        real_orbit=True,
    )
