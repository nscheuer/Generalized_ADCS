from dataclasses import dataclass
from functools import lru_cache
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ADCS.CONOPS.goals import Coordinate_Goal, ECI_Goal, Goal, No_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State


POINTING_CFG = {
    "p_gain": 5.0e-5,
    "d_gain": 1.0e-3,
    "c_gain": 0.0,
    "h_target": np.array([0.004, 0.0, 0.0]),
}
DESAT_CFG = {
    "p_gain": 5.0e-5,
    "d_gain": 5.0e-5,
    "c_gain": 2.0e-2,
    "h_target": np.array([0.002, 0.0, 0.0]),
}
CTRL_EFFORT_TOL = 0.01

SCENARIO_DEFAULTS = {
    "hold": (500.0, 2.0),
    "easy_turn": (500.0, 2.0),
    "hard_turn": (500.0, 2.0),
    "ground_tracking": (500.0, 2.0),
    "desat": (2500.0, 5.0),
}


@dataclass(frozen=True)
class MTQwRWLPRun:
    time_hist: np.ndarray
    state_hist: np.ndarray
    os_hist: list[Orbital_State]
    sensor_hist: np.ndarray
    u_hist: np.ndarray
    boresight_hist: np.ndarray
    goal: Goal


class StaticGoal(Goal):
    def __init__(
        self,
        *,
        q_err: np.ndarray | None = None,
        goal_vec_eci: np.ndarray | None = None,
        w_ref_eci: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.q_err = np.zeros(3) if q_err is None else np.asarray(q_err, dtype=float)
        self.goal_vec_eci = (
            np.array([0.0, 0.0, 1.0])
            if goal_vec_eci is None
            else normalize(np.asarray(goal_vec_eci, dtype=float))
        )
        self.w_ref_eci = np.zeros(3) if w_ref_eci is None else np.asarray(w_ref_eci, dtype=float)
        self.boresight_name = None

    def to_ref(self, os0: Orbital_State) -> tuple[np.ndarray, np.ndarray]:
        goal = np.empty(4)
        goal[0] = np.nan
        goal[1:] = self.goal_vec_eci
        return goal, self.w_ref_eci

    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        return self.q_err.copy()


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _make_satellite(
    *,
    rw_axes: list[np.ndarray] | None = None,
    mtq_max_torque: float = 0.4,
    rw_max_torque: float = 7.0e-3,
    rw_h: float | list[float] | np.ndarray = 5.0e-3,
) -> Satellite:
    mtqs = [MTQ(axis=axis, max_torque=mtq_max_torque) for axis in MathConstants.unitvecs]
    actuators = list(mtqs)

    axes = [MathConstants.unitvecs[0]] if rw_axes is None else rw_axes
    rw_h_array = np.repeat(float(rw_h), len(axes)) if np.isscalar(rw_h) else np.asarray(rw_h, dtype=float)
    for index, axis in enumerate(axes):
        actuators.append(RW(axis=axis, max_torque=rw_max_torque, J=1.0e-3, h=rw_h_array[index], h_max=16.2e-3))

    mtms = [MTM(axis=axis) for axis in MathConstants.unitvecs]
    return Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=actuators,
        sensors=mtms,
        boresight=np.array([0.0, 0.0, 1.0]),
    )


def _controller_cfg(goal: Goal) -> dict:
    return DESAT_CFG if isinstance(goal, No_Goal) else POINTING_CFG


def _make_controller(est_sat: Satellite, goal: Goal) -> MTQ_w_RW_LP:
    cfg = _controller_cfg(goal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MTQ_w_RW_LP(
            est_sat=est_sat,
            p_gain=cfg["p_gain"],
            d_gain=cfg["d_gain"],
            c_gain=cfg["c_gain"],
            h_target=cfg["h_target"],
        )


@lru_cache(maxsize=16)
def _make_real_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    os0 = Orbital_State(
        ephem=ephem,
        J2000=start_time,
        R=7000.0 * np.array([0.0, np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]),
        V=np.array([8.0, 0.0, 0.0]),
    )
    return Orbit(os0=os0, end_time=end_time, dt=dt, zonal_J=2, fast=False)


def _make_fake_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - TimeConstants.sec2cent,
        R=7000.0 * np.array([0.0, np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1.0e5 + 1.0, 0.0, 0.0]),
        rho=5.0e-12,
    )
    count = int(tf / dt) + 20
    orbit_states = []
    for step in range(count):
        current = os0.copy()
        current.J2000 = os0.J2000 + step * dt * TimeConstants.sec2cent
        orbit_states.append(current)
    return Orbit(orbit_states)


def _scenario_goal(name: str) -> Goal:
    match name:
        case "hold":
            return ECI_Goal(np.array([0.0, 0.0, 1.0]))
        case "easy_turn":
            return ECI_Goal(np.array([0.0, 1.0, 0.0]))
        case "hard_turn":
            return ECI_Goal(np.array([1.0, 1.0, 1.0]))
        case "ground_tracking":
            return Coordinate_Goal(lat=40, lon=-40, alt=0)
        case "desat":
            return No_Goal()
        case _:
            raise ValueError(f"Unknown scenario '{name}'.")


def _initial_state() -> State:
    rng_state = np.random.get_state()
    np.random.seed(1)
    try:
        w0 = random_n_unit_vec(3) * np.random.uniform(1.0, 2.0) * np.pi / 180.0
        w0 = np.zeros(3)
        q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
        return State(w=w0, q=q0, h=[5.0e-3])
    finally:
        np.random.set_state(rng_state)


def run_mtq_w_rw_lp_simulation(
    scenario_name: str,
    *,
    tf: float | None = None,
    dt: float | None = None,
    real_orbit: bool = True,
    progress: bool = False,
) -> MTQwRWLPRun:
    goal = _scenario_goal(scenario_name)
    default_tf, default_dt = SCENARIO_DEFAULTS[scenario_name]
    horizon = default_tf if tf is None else tf
    step = default_dt if dt is None else dt

    satellite = _make_satellite()
    controller = _make_controller(satellite, goal)
    orbit = _make_real_orbit(horizon, step) if real_orbit else _make_fake_orbit(horizon, step)
    x = _initial_state()

    steps = int(horizon / step)
    time_hist = np.full(steps, np.nan)
    state_hist = np.full((steps, x.as_array().size), np.nan)
    sensor_hist = np.full((steps, len(satellite.sensors + satellite.rw_actuators)), np.nan)
    u_hist = np.full((steps, len(satellite.actuators)), np.nan)
    boresight_hist = np.full((steps, 4), np.nan)
    os_hist: list[Orbital_State] = []

    iterator = tqdm(range(steps), desc=f"Simulating {scenario_name}") if progress else range(steps)
    t = 0.0
    for index in iterator:
        os_now = orbit.get_os(J2000=0.22 + t * TimeConstants.sec2cent)
        sens = satellite.sensor_readings(x=x, os=os_now)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=satellite, os_hat=os_now, goal=goal)

        time_hist[index] = t
        state_hist[index, :] = x.as_array()
        sensor_hist[index, :] = sens
        u_hist[index, :] = u
        boresight_hist[index, :] = goal.to_ref(os0=os_now)[0]
        os_hist.append(os_now)

        next_t = t + step
        os_next = orbit.get_os(J2000=0.22 + next_t * TimeConstants.sec2cent)
        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0.0, step),
            y0=x.as_array(),
            method="RK45",
            args=(u, os_now, os_next),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        x = State.from_array(out.y[:, -1]).normalized()
        t = next_t

    return MTQwRWLPRun(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        sensor_hist=sensor_hist,
        u_hist=u_hist,
        boresight_hist=boresight_hist,
        goal=goal,
    )


def _quat_to_dcm_body_to_eci(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2)],
        ]
    )


def _final_pointing_error_deg(run: MTQwRWLPRun) -> float:
    valid = np.where(np.isfinite(run.state_hist[:, 0]))[0]
    last = valid[-1]
    q = run.state_hist[last, 3:7]
    goal_eci = run.boresight_hist[last, 1:4]
    boresight_eci = _quat_to_dcm_body_to_eci(q) @ np.array([0.0, 0.0, 1.0])
    boresight_eci = boresight_eci / (np.linalg.norm(boresight_eci) + 1.0e-16)
    goal_eci = goal_eci / (np.linalg.norm(goal_eci) + 1.0e-16)
    return float(np.degrees(np.arccos(np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0))))


def _final_control_effort(run: MTQwRWLPRun) -> float:
    valid = np.where(np.isfinite(run.u_hist[:, 0]))[0]
    return float(np.linalg.norm(run.u_hist[valid[-1], :]))


def _final_rate_norm(run: MTQwRWLPRun) -> float:
    valid = np.where(np.isfinite(run.state_hist[:, 0]))[0]
    return float(np.linalg.norm(run.state_hist[valid[-1], 0:3]))


def _achieved_torque(
    satellite: Satellite,
    u_rw: np.ndarray,
    u_mtq: np.ndarray,
    b_body: np.ndarray,
) -> np.ndarray:
    rws = [actuator for actuator in satellite.actuators if isinstance(actuator, RW)]
    mtqs = [actuator for actuator in satellite.actuators if isinstance(actuator, MTQ)]
    tau_rw = sum(np.asarray(rw.axis, dtype=float) * u_rw[index] for index, rw in enumerate(rws)) if rws else np.zeros(3)
    tau_mtq = sum(
        np.cross(np.asarray(mtq.axis, dtype=float) * u_mtq[index], b_body)
        for index, mtq in enumerate(mtqs)
    ) if mtqs else np.zeros(3)
    return tau_rw + tau_mtq


@pytest.fixture
def base_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([2.0e-5, -3.0e-5, 4.0e-5]),
        S=np.array([1.0e5, 0.0, 0.0]),
        rho=5.0e-12,
    )


@pytest.fixture
def lp_satellite() -> Satellite:
    return _make_satellite(rw_axes=[MathConstants.unitvecs[0]])


@pytest.fixture
def lp_controller(lp_satellite: Satellite) -> MTQ_w_RW_LP:
    return _make_controller(lp_satellite, ECI_Goal(np.array([0.0, 0.0, 1.0])))


@pytest.fixture
def desat_controller(lp_satellite: Satellite) -> MTQ_w_RW_LP:
    return _make_controller(lp_satellite, No_Goal())


@pytest.fixture
def base_state() -> State:
    q = normalize(np.array([0.85, 0.1, -0.15, 0.3]))
    return State(w=[0.02, -0.015, 0.01], q=q, h=[0.006])


def test_mtq_w_rw_lp_none_goal_routes_to_desaturation(
    lp_satellite: Satellite,
    desat_controller: MTQ_w_RW_LP,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    sens = lp_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command_none = desat_controller.find_u(
        x_hat=base_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=None,
    )
    command_no_goal = desat_controller.find_u_desaturate(
        x_hat=base_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command_none, command_no_goal)


def test_mtq_w_rw_lp_pointing_goal_routes_to_pointing_mode(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    sens = lp_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command = lp_controller.find_u(
        x_hat=base_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = lp_controller.find_u_pointing(
        x_hat=base_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )

    np.testing.assert_allclose(command, expected)


def test_allocate_max_torque_returns_zero_for_zero_request(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
    base_orbital_state: Orbital_State,
) -> None:
    u_rw, u_mtq, alpha = lp_controller.allocate_max_torque_in_direction(
        np.zeros(3),
        base_orbital_state.B,
        lp_satellite,
    )

    np.testing.assert_allclose(u_rw, np.zeros(1))
    np.testing.assert_allclose(u_mtq, np.zeros(3))
    assert alpha == 1.0


def test_allocate_max_torque_uses_rw_exactly_along_rw_axis(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
) -> None:
    tau_des = np.array([3.0e-3, 0.0, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = lp_controller.allocate_max_torque_in_direction(tau_des, b_body, lp_satellite)
    achieved = _achieved_torque(lp_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(achieved, tau_des, atol=1.0e-10)
    np.testing.assert_array_less(np.abs(u_rw), lp_controller.rw_umax + 1.0e-12)
    np.testing.assert_array_less(np.abs(u_mtq), lp_controller.mtq_umax + 1.0e-12)
    assert alpha == 1.0


def test_allocate_max_torque_scales_when_request_exceeds_rw_limit(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
) -> None:
    tau_des = np.array([2.0e-2, 0.0, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = lp_controller.allocate_max_torque_in_direction(tau_des, b_body, lp_satellite)
    achieved = _achieved_torque(lp_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_array_less(np.abs(u_rw), lp_controller.rw_umax + 1.0e-12)
    np.testing.assert_array_less(np.abs(u_mtq), lp_controller.mtq_umax + 1.0e-12)
    np.testing.assert_allclose(achieved, alpha * tau_des, atol=1.0e-10)
    assert 0.0 < alpha < 1.0


def test_allocate_max_torque_returns_zero_for_mtq_parallel_field_request(
    base_orbital_state: Orbital_State,
) -> None:
    satellite = _make_satellite(rw_axes=[])
    controller = _make_controller(satellite, ECI_Goal(np.array([0.0, 0.0, 1.0])))
    b_body = np.array([0.0, 0.0, 1.0])
    tau_des = np.array([0.0, 0.0, 2.0e-3])

    u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(tau_des, b_body, satellite)
    achieved = _achieved_torque(satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(u_rw, np.zeros(0))
    np.testing.assert_allclose(achieved, np.zeros(3), atol=1.0e-12)
    np.testing.assert_array_less(np.abs(u_mtq), controller.mtq_umax + 1.0e-12)
    assert alpha == 0.0


def test_allocate_max_torque_scales_perpendicular_mtq_request_to_available_torque(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
) -> None:
    tau_des = np.array([0.0, 1.5e-3, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = lp_controller.allocate_max_torque_in_direction(tau_des, b_body, lp_satellite)
    achieved = _achieved_torque(lp_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(achieved, alpha * tau_des, atol=1.0e-10)
    np.testing.assert_array_less(np.abs(u_rw), lp_controller.rw_umax + 1.0e-12)
    np.testing.assert_array_less(np.abs(u_mtq), lp_controller.mtq_umax + 1.0e-12)
    assert 0.0 < alpha < 1.0


def test_pointing_mode_matches_primary_allocation_when_secondary_gain_is_zero(
    lp_satellite: Satellite,
    lp_controller: MTQ_w_RW_LP,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    goal = StaticGoal(q_err=np.array([0.2, -0.1, 0.05]), w_ref_eci=np.array([0.01, 0.0, -0.02]))
    sens = lp_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command = lp_controller.find_u_pointing(
        x_hat=base_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )

    w = base_state.w
    q = base_state.q
    h_rw_body = np.array([base_state.h[0], 0.0, 0.0])
    w_ref_body = rot_mat(q).T @ goal.w_ref_eci
    tau_pd = -lp_controller.p_gain * goal.q_err - lp_controller.d_gain * (w - w_ref_body)
    tau_gyro = np.cross(w, lp_satellite.J_0 @ w + h_rw_body)
    tau_des = tau_pd + tau_gyro
    b_body = lp_controller.M_mtm_read @ sens
    u_rw, u_mtq, _ = lp_controller.allocate_max_torque_in_direction(tau_des, b_body, lp_satellite)
    expected = np.zeros(len(lp_satellite.actuators))
    expected[lp_controller.rw_indices] = u_rw
    expected[lp_controller.mtq_indices] = u_mtq

    np.testing.assert_allclose(command, expected)


def test_pointing_mode_with_zero_field_and_no_rws_achieves_zero_torque(
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    satellite = _make_satellite(rw_axes=[])
    controller = _make_controller(satellite, ECI_Goal(np.array([0.0, 0.0, 1.0])))
    goal = ECI_Goal(np.array([0.0, 0.0, 1.0]))
    zero_sens = np.zeros(len(satellite.sensors + satellite.rw_actuators))
    command = controller.find_u_pointing(
        x_hat=State(w=base_state.w, q=base_state.q),
        sens=zero_sens,
        est_sat=satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    achieved = _achieved_torque(satellite, np.zeros(0), command, np.zeros(3))

    np.testing.assert_allclose(achieved, np.zeros(3), atol=1.0e-12)
    np.testing.assert_array_less(np.abs(command), controller.mtq_umax + 1.0e-12)


def test_desaturate_mode_zero_field_returns_zero_output(
    lp_satellite: Satellite,
    desat_controller: MTQ_w_RW_LP,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    command = desat_controller.find_u_desaturate(
        x_hat=base_state,
        sens=np.zeros(len(lp_satellite.sensors + lp_satellite.rw_actuators)),
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command, np.zeros(len(lp_satellite.actuators)))


def test_desaturate_mode_ignores_nan_sensor_values(
    lp_satellite: Satellite,
    desat_controller: MTQ_w_RW_LP,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    sens = lp_satellite.sensor_readings(x=base_state, os=base_orbital_state)
    sens_with_nan = sens.copy()
    sens_with_nan[0] = np.nan

    command_nan = desat_controller.find_u_desaturate(
        x_hat=base_state,
        sens=sens_with_nan,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    command_zeroed = desat_controller.find_u_desaturate(
        x_hat=base_state,
        sens=np.nan_to_num(sens_with_nan, nan=0.0),
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command_nan, command_zeroed)


def test_desaturate_mode_without_rw_state_uses_zero_rw_momentum(
    lp_satellite: Satellite,
    desat_controller: MTQ_w_RW_LP,
    base_orbital_state: Orbital_State,
) -> None:
    short_state = State(w=[0.02, -0.015, 0.01], q=[1.0, 0.0, 0.0, 0.0])
    full_state = State(w=short_state.w, q=short_state.q, h=[0.0])
    sens = lp_satellite.sensor_readings(x=full_state, os=base_orbital_state)

    command_short = desat_controller.find_u_desaturate(
        x_hat=short_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    command_zero_momentum = desat_controller.find_u_desaturate(
        x_hat=full_state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command_short, command_zero_momentum)


def test_desaturate_mode_respects_rw_and_mtq_limits(
    lp_satellite: Satellite,
    desat_controller: MTQ_w_RW_LP,
    base_orbital_state: Orbital_State,
) -> None:
    state = State(w=[4.0, -3.0, 2.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.02])
    sens = np.array([5.0e-5, -3.0e-5, 2.0e-5, 0.0])

    command = desat_controller.find_u_desaturate(
        x_hat=state,
        sens=sens,
        est_sat=lp_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_array_less(np.abs(command[desat_controller.mtq_indices]), desat_controller.mtq_umax + 1.0e-12)
    np.testing.assert_array_less(np.abs(command[desat_controller.rw_indices]), desat_controller.rw_umax + 1.0e-12)



@pytest.mark.parametrize(
    ("scenario_name", "pointing_tol_deg", "control_tol", "rate_tol"),
    [
        ("hold", 0.1, CTRL_EFFORT_TOL, None),
        ("hard_turn", 0.5, CTRL_EFFORT_TOL, None),
        ("ground_tracking", 5.0, None, None),
        ("desat", None, CTRL_EFFORT_TOL, 1.0e-4),
    ],
)
def test_mtq_w_rw_lp_converges_in_representative_scenarios(
    scenario_name: str,
    pointing_tol_deg: float | None,
    control_tol: float | None,
    rate_tol: float | None,
) -> None:
    run = run_mtq_w_rw_lp_simulation(scenario_name)

    if pointing_tol_deg is not None:
        assert _final_pointing_error_deg(run) <= pointing_tol_deg
    if control_tol is not None:
        assert _final_control_effort(run) <= control_tol
    if rate_tol is not None:
        assert _final_rate_norm(run) <= rate_tol


def debug_plots(run: MTQwRWLPRun) -> None:
    from ADCS.CONOPS.goals import Coordinate_Goal as CoordGoal
    from ADCS.helpers.plotting.animate_estimator import animate_attitude
    from ADCS.helpers.plotting.animate_orbit_pyvista import animate_orbit_pyvista
    from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
    from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking
    from ADCS.helpers.plotting.plot_estimator import plot_state_comparison

    animate_attitude(
        time=run.time_hist,
        state_hist=run.state_hist,
        os_hist=run.os_hist,
        boresight_goal_hist=run.boresight_hist,
    )
    plot_control(time=run.time_hist, u_hist=run.u_hist)
    plot_state_comparison(time=run.time_hist, state_hist=run.state_hist)
    plot_rw_momentum(time=run.time_hist, state_hist=run.state_hist)
    animate_orbit_pyvista(
        time_hist=run.time_hist,
        state_hist=run.state_hist,
        os_hist=run.os_hist,
        boresight_goal_hist=run.boresight_hist,
        coord_goal=run.goal if isinstance(run.goal, CoordGoal) else None,
    )
    plot_target_tracking(
        state_hist=run.state_hist,
        boresight_hist=run.boresight_hist,
        body_boresight=np.array([0.0, 0.0, 1.0]),
    )
    create_close_all_button_window()


def main() -> None:
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "desat"
    if scenario_name not in SCENARIO_DEFAULTS:
        raise SystemExit(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIO_DEFAULTS)}")

    run = run_mtq_w_rw_lp_simulation(scenario_name, progress=True, real_orbit=True)
    debug_plots(run)


if __name__ == "__main__":
    main()
