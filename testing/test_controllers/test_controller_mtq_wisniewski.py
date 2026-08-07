from dataclasses import dataclass
from functools import lru_cache
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ADCS.CONOPS.goals import ECI_Goal, Goal, No_Goal
from ADCS.controller import MTQ_Wisniewski
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.helpers.math_helpers import rot_mat
from ADCS.helpers.math_helpers import Wmat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State


LAMBDA_S = np.diag([0.003, 0.003, 0.003])
LAMBDA_Q = np.diag([0.002, 0.002, 0.002])
DT_PHYSICS = 50.0
TF = 10000.0

SCENARIO_NAMES = [
    "stop_rot_zero",
    "stop_rot_moving",
    "stop_rot_moving_wh",
    "align_x_zero",
    "align_x_zero_wh",
    "align_x_moving",
    "align_x_moving_wh",
    "align_xyz_zero",
    "align_xyz_zero_wh",
    "align_xyz_moving",
    "align_xyz_moving_wh",
]


@dataclass(frozen=True)
class WisniewskiRun:
    time_hist: np.ndarray
    state_hist: list[State]
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

    def error(
        self,
        q: np.ndarray,
        body_boresight: np.ndarray,
        os0: Orbital_State,
    ) -> np.ndarray:
        return self.q_err.copy()


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _mtq_indices(satellite: Satellite) -> list[int]:
    return [
        index for index, actuator in enumerate(satellite.actuators) if isinstance(actuator, MTQ)
    ]


def _make_satellite(
    *,
    include_rw: bool = True,
    mtq_max_torque: float = 1.0,
    initial_rw_h: float | np.ndarray = 0.0,
) -> Satellite:
    mtqs = [MTQ(axis=axis, max_torque=mtq_max_torque) for axis in MathConstants.unitvecs]
    actuators = list(mtqs)

    if include_rw:
        h0 = (
            np.repeat(float(initial_rw_h), 3)
            if np.isscalar(initial_rw_h)
            else np.asarray(initial_rw_h, dtype=float)
        )
        rws = [
            RW(axis=axis, max_torque=7.0e-3, J=1.0e-3, h=h0[i], h_max=16.2e-3)
            for i, axis in enumerate(MathConstants.unitvecs)
        ]
        actuators.extend(rws)

    mtms = [MTM(axis=axis) for axis in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=actuators,
        sensors=mtms,
        boresight=np.array([0.0, 0.0, 1.0]),
    )


def _make_controller(est_sat: Satellite) -> MTQ_Wisniewski:
    return MTQ_Wisniewski(est_sat=est_sat, lambda_s=LAMBDA_S, lambda_q=LAMBDA_Q)


@lru_cache(maxsize=4)
def _make_real_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    os0 = Orbital_State(
        ephem=ephem,
        J2000=start_time,
        R=7000.0 * np.array([0.0, -np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]),
        V=np.array([8.0, 0.0, 0.0]),
    )
    return Orbit(os0=os0, end_time=end_time, dt=dt, zonal_J=2, fast=False)


def _build_scenario(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Goal, float]:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{name}'.")

    w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.zeros(3)
    goal: Goal = No_Goal()
    tf = TF

    if "moving" in name:
        rng_state = np.random.get_state()
        np.random.seed(37)
        try:
            scale = 1.5 if "stop" in name else 0.2
            w0 = random_n_unit_vec(3) * np.random.uniform(scale * 0.5, scale * 0.8) * np.pi / 180.0
            q0 = normalize(random_n_unit_vec(4))
        finally:
            np.random.set_state(rng_state)

    if "wh" in name:
        h0 = np.full(3, 1.0e-3)

    if "align_x" in name:
        goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    elif "align_xyz" in name:
        goal = ECI_Goal(_unit(np.array([1.0, 1.0, 1.0])))

    return w0, q0, h0, goal, tf


def _expected_command(
    controller: MTQ_Wisniewski,
    satellite: Satellite,
    x_hat: State,
    sens: np.ndarray,
    os_hat: Orbital_State,
    goal: Goal | None,
) -> np.ndarray:
    active_goal = No_Goal() if goal is None else goal
    w = x_hat.w
    q = x_hat.q

    rw_actuators = [actuator for actuator in satellite.actuators if isinstance(actuator, RW)]
    h_rw_states = controller.reaction_wheel_momentum_states(
        x_hat,
        len(rw_actuators),
        context="test expected MTQ_Wisniewski",
    )

    _, w_ref_eci = active_goal.to_ref(os0=os_hat)
    w_ref_body = rot_mat(q).T @ w_ref_eci
    q_err = active_goal.error(
        q=q,
        body_boresight=satellite.get_boresight(active_goal.boresight_name),
        os0=os_hat,
    )
    w_err = w - w_ref_body

    sliding_surface = satellite.J_0 @ w_err + controller.lambda_q @ q_err

    h_rw_body = np.zeros(3)
    for h_rw, actuator in zip(h_rw_states, rw_actuators):
        h_rw_body += np.asarray(actuator.axis, dtype=float).reshape(-1) * h_rw

    q_err_full = np.hstack(([np.sqrt(max(0.0, 1.0 - np.dot(q_err, q_err)))], q_err))
    q_err_dot = 0.5 * w_err @ Wmat(q_err_full).T

    tau_des = (
        np.cross(w, satellite.J_0 @ w + h_rw_body)
        + satellite.J_0 @ np.cross(w, w_err)
        - controller.lambda_q @ q_err_dot[1:4]
        - controller.lambda_s @ sliding_surface
    )

    sens_clean = np.asarray(sens, dtype=float).reshape(-1).copy()
    sens_clean[np.isnan(sens_clean)] = 0.0
    b_body = controller.M_read @ sens_clean
    b_norm_sq = np.linalg.norm(b_body) ** 2

    if b_norm_sq < 1.0e-11:
        u_mtq = np.zeros(3)
    else:
        u_mtq = np.cross(b_body, tau_des) / b_norm_sq

    ratios = np.where(np.abs(u_mtq) > 0.0, controller.mtq_umax / np.abs(u_mtq), np.inf)
    u_mtq *= min(1.0, float(np.min(ratios)))

    command = np.zeros(len(satellite.actuators))
    command[_mtq_indices(satellite)] = u_mtq
    return command


def run_mtq_wisniewski_simulation(
    scenario_name: str,
    *,
    tf: float | None = None,
    dt: float = DT_PHYSICS,
    progress: bool = False,
) -> WisniewskiRun:
    w0, q0, h0, goal, default_tf = _build_scenario(scenario_name)
    horizon = default_tf if tf is None else tf

    satellite = _make_satellite(include_rw=True, initial_rw_h=h0)
    controller = _make_controller(satellite)
    orbit = _make_real_orbit(horizon, dt)
    x = State(w=w0, q=q0, h=h0)

    steps = int(horizon / dt)
    time_hist = np.full(steps, np.nan)
    state_hist: list[State] = []
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
        state_hist.append(x.copy())
        sensor_hist[index, :] = sens
        u_hist[index, :] = u
        boresight_hist[index, :] = goal.to_ref(os0=os_now)[0]
        os_hist.append(os_now)

        next_t = t + dt
        os_next = orbit.get_os(J2000=0.22 + next_t * TimeConstants.sec2cent)
        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0.0, dt),
            y0=x.as_array(),
            method="RK45",
            args=(u, os_now, os_next),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        x = State.from_array(out.y[:, -1])
        x = x.normalized()
        t = next_t

    return WisniewskiRun(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        sensor_hist=sensor_hist,
        u_hist=u_hist,
        boresight_hist=boresight_hist,
        goal=goal,
    )


def _final_alignment_error_deg(run: WisniewskiRun) -> float:
    q_final = run.state_hist[-1].q
    goal_vec = run.boresight_hist[len(run.state_hist) - 1, 1:4]
    err_vec = vector_alignment_error(q_final, goal_vec, np.array([0.0, 0.0, 1.0]))
    return float(np.degrees(np.linalg.norm(err_vec)))


def _final_rate_norm(run: WisniewskiRun) -> float:
    return float(np.linalg.norm(run.state_hist[-1].w))


@pytest.fixture
def wisniewski_satellite() -> Satellite:
    return _make_satellite(include_rw=False, mtq_max_torque=1.0)


@pytest.fixture
def wisniewski_satellite_with_rw() -> Satellite:
    return _make_satellite(include_rw=True, mtq_max_torque=1.0, initial_rw_h=np.array([0.004, -0.002, 0.003]))


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
def no_rw_state() -> State:
    q = normalize(np.array([0.9, 0.1, -0.2, 0.3]))
    w = np.array([0.03, -0.02, 0.01])
    return State(w=w, q=q)


@pytest.fixture
def rw_state() -> State:
    q = normalize(np.array([0.8, -0.1, 0.2, 0.55]))
    w = np.array([0.025, -0.015, 0.02])
    h = np.array([0.006, -0.004, 0.003])
    return State(w=w, q=q, h=h)


def test_mtq_wisniewski_none_goal_matches_no_goal(
    wisniewski_satellite: Satellite,
    no_rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    sens = wisniewski_satellite.sensor_readings(x=no_rw_state, os=base_orbital_state)

    command_none = controller.find_u(
        x_hat=no_rw_state,
        sens=sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=None,
    )
    command_no_goal = controller.find_u(
        x_hat=no_rw_state,
        sens=sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command_none, command_no_goal)


def test_mtq_wisniewski_zero_field_returns_zero_command(
    wisniewski_satellite: Satellite,
    no_rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    zero_sens = np.zeros(len(wisniewski_satellite.sensors))

    command = controller.find_u(
        x_hat=no_rw_state,
        sens=zero_sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command, np.zeros(len(wisniewski_satellite.actuators)))


def test_mtq_wisniewski_tiny_field_returns_zero_command(
    wisniewski_satellite: Satellite,
    no_rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    tiny_sens = np.array([1.0e-7, -1.0e-7, 2.0e-7])

    command = controller.find_u(
        x_hat=no_rw_state,
        sens=tiny_sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command, np.zeros(len(wisniewski_satellite.actuators)))


def test_mtq_wisniewski_only_commands_mtqs_when_reaction_wheels_present(
    wisniewski_satellite_with_rw: Satellite,
    rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite_with_rw)
    sens = wisniewski_satellite_with_rw.sensor_readings(x=rw_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=rw_state,
        sens=sens,
        est_sat=wisniewski_satellite_with_rw,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command[3:], np.zeros(3))
    assert np.any(np.abs(command[:3]) > 0.0)


def test_mtq_wisniewski_cleans_nan_sensor_values(
    wisniewski_satellite: Satellite,
    no_rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    sens = wisniewski_satellite.sensor_readings(x=no_rw_state, os=base_orbital_state)
    sens_with_nan = sens.copy()
    sens_with_nan[1] = np.nan

    command = controller.find_u(
        x_hat=no_rw_state,
        sens=sens_with_nan,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    expected = _expected_command(
        controller,
        wisniewski_satellite,
        no_rw_state,
        sens_with_nan,
        base_orbital_state,
        No_Goal(),
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_wisniewski_matches_sliding_law_without_goal(
    wisniewski_satellite: Satellite,
    no_rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    sens = wisniewski_satellite.sensor_readings(x=no_rw_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=no_rw_state,
        sens=sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    expected = _expected_command(
        controller,
        wisniewski_satellite,
        no_rw_state,
        sens,
        base_orbital_state,
        No_Goal(),
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_wisniewski_matches_eci_goal_alignment_law(
    wisniewski_satellite: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    state = State(w=[0.01, -0.015, 0.02], q=[1.0, 0.0, 0.0, 0.0])
    goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    sens = wisniewski_satellite.sensor_readings(x=state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = _expected_command(
        controller,
        wisniewski_satellite,
        state,
        sens,
        base_orbital_state,
        goal,
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_wisniewski_rotates_reference_rate_into_body_frame(
    wisniewski_satellite: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite)
    q = normalize(np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    state = State(w=[0.03, 0.02, -0.01], q=q)
    goal = StaticGoal(q_err=np.zeros(3), w_ref_eci=np.array([0.04, 0.0, 0.0]))
    sens = wisniewski_satellite.sensor_readings(x=state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=wisniewski_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = _expected_command(
        controller,
        wisniewski_satellite,
        state,
        sens,
        base_orbital_state,
        goal,
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_wisniewski_uses_rw_momentum_from_state_when_available(
    wisniewski_satellite_with_rw: Satellite,
    rw_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite_with_rw)
    sens = wisniewski_satellite_with_rw.sensor_readings(x=rw_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=rw_state,
        sens=sens,
        est_sat=wisniewski_satellite_with_rw,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    expected_from_state = _expected_command(
        controller,
        wisniewski_satellite_with_rw,
        rw_state,
        sens,
        base_orbital_state,
        No_Goal(),
    )
    expected_from_actuators = _expected_command(
        controller,
        wisniewski_satellite_with_rw,
        State(
            w=rw_state.w,
            q=rw_state.q,
            h=[actuator.h for actuator in wisniewski_satellite_with_rw.actuators if isinstance(actuator, RW)],
        ),
        sens,
        base_orbital_state,
        No_Goal(),
    )

    np.testing.assert_allclose(command, expected_from_state)
    assert not np.allclose(expected_from_state, expected_from_actuators)


def test_mtq_wisniewski_rejects_missing_rw_momentum_state(
    wisniewski_satellite_with_rw: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite_with_rw)
    short_state = State(w=[0.015, -0.01, 0.025], q=[1.0, 0.0, 0.0, 0.0])
    sens = wisniewski_satellite_with_rw.sensor_readings(
        x=State(w=short_state.w, q=short_state.q, h=np.zeros(3)),
        os=base_orbital_state,
    )

    with pytest.raises(ValueError, match="exactly 3 reaction-wheel momentum states, got 0"):
        controller.find_u(
            x_hat=short_state,
            sens=sens,
            est_sat=wisniewski_satellite_with_rw,
            os_hat=base_orbital_state,
            goal=No_Goal(),
        )


def test_mtq_wisniewski_rejects_extra_rw_momentum_state(
    wisniewski_satellite_with_rw: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(wisniewski_satellite_with_rw)
    state = State(w=[0.015, -0.01, 0.025], q=[1.0, 0.0, 0.0, 0.0], h=[0.0, 0.0, 0.0, 0.0])
    sens = wisniewski_satellite_with_rw.sensor_readings(
        x=State(w=state.w, q=state.q, h=np.zeros(3)),
        os=base_orbital_state,
    )

    with pytest.raises(ValueError, match="exactly 3 reaction-wheel momentum states, got 4"):
        controller.find_u(
            x_hat=state,
            sens=sens,
            est_sat=wisniewski_satellite_with_rw,
            os_hat=base_orbital_state,
            goal=No_Goal(),
        )


def test_mtq_wisniewski_saturates_with_uniform_scaling(
    base_orbital_state: Orbital_State,
) -> None:
    satellite = _make_satellite(include_rw=False, mtq_max_torque=0.01)
    controller = _make_controller(satellite)
    state = State(w=[4.0, -3.0, 2.0], q=[1.0, 0.0, 0.0, 0.0])
    goal = StaticGoal(q_err=np.array([0.8, -0.5, 0.75]), w_ref_eci=np.array([0.6, -0.1, 0.2]))
    sens = np.array([1.5e-5, -1.0e-5, 2.0e-5])

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = _expected_command(
        controller,
        satellite,
        state,
        sens,
        base_orbital_state,
        goal,
    )

    np.testing.assert_allclose(command, expected)
    assert np.isclose(np.max(np.abs(command[:3])), 0.01)



@pytest.mark.parametrize(
    ("scenario_name", "alignment_tol_deg"),
    [
        ("stop_rot_moving", None),
        ("align_x_moving", 5.0),
        ("align_xyz_moving", 5.0),
        ("align_xyz_moving_wh", 10.0),
    ],
)
def test_mtq_wisniewski_converges_in_representative_scenarios(
    scenario_name: str,
    alignment_tol_deg: float | None,
) -> None:
    run = run_mtq_wisniewski_simulation(scenario_name)

    assert _final_rate_norm(run) < 2.0e-3

    if alignment_tol_deg is not None:
        assert _final_alignment_error_deg(run) < alignment_tol_deg


def debug_plots(run: WisniewskiRun) -> None:
    from ADCS.helpers.plotting.animate_estimator import animate_attitude
    from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
    from ADCS.helpers.plotting.plot_controller import (
        plot_control,
        plot_rw_momentum,
        plot_target_tracking,
    )
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
    plot_target_tracking(
        state_hist=run.state_hist,
        boresight_hist=run.boresight_hist,
        body_boresight=np.array([0.0, 0.0, 1.0]),
    )
    create_close_all_button_window()


def main() -> None:
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "align_xyz_zero"
    if scenario_name not in SCENARIO_NAMES:
        raise SystemExit(f"Unknown scenario '{scenario_name}'. Available: {SCENARIO_NAMES}")

    run = run_mtq_wisniewski_simulation(scenario_name, progress=True)
    debug_plots(run)


if __name__ == "__main__":
    main()
