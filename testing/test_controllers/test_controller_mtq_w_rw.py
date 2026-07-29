from dataclasses import dataclass
from functools import lru_cache
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ADCS.CONOPS.goals import Coordinate_Goal, ECI_Goal, Goal, No_Goal
from ADCS.controller import MTQ_w_RW
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import limit
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.helpers.math_helpers import rot_mat
from ADCS.helpers.math_helpers import skewsym
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State


STOP_CFG = {"p_gain": 0.0, "d_gain": 1.0, "c_gain": 0.0}
POINTING_CFG = {"p_gain": 0.1, "d_gain": 0.7, "c_gain": 0.0}
FULL_CFG = {"p_gain": 0.1, "d_gain": 0.7, "c_gain": 0.1}

SCENARIO_DEFAULTS = {
    "stop_rotation": (100.0, 1.0, False),
    "align_x": (100.0, 1.0, False),
    "align_diag": (100.0, 1.0, False),
    "desaturate": (100.0, 1.0, False),
    "full_task": (100.0, 1.0, True),
}


@dataclass(frozen=True)
class MTQwRWRun:
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


def _make_satellite(
    *,
    mtq_max_torque: float = 0.1,
    rw_max_torque: float = 4.51,
    rw_h: float | np.ndarray = 0.0,
    boresight: np.ndarray | None = None,
) -> Satellite:
    mtqs = [MTQ(axis=axis, max_torque=mtq_max_torque) for axis in MathConstants.unitvecs]
    rw_h_array = np.repeat(float(rw_h), 3) if np.isscalar(rw_h) else np.asarray(rw_h, dtype=float)
    rws = [
        RW(axis=axis, max_torque=rw_max_torque, J=0.22, h=rw_h_array[index], h_max=3.8)
        for index, axis in enumerate(MathConstants.unitvecs)
    ]
    mtms = [MTM(axis=axis) for axis in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0.0, 0.0, 1.0]) if boresight is None else np.asarray(boresight, dtype=float),
    )


def _make_controller(est_sat: Satellite, *, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray | None = None) -> MTQ_w_RW:
    target = np.zeros(3) if h_target is None else np.asarray(h_target, dtype=float)
    return MTQ_w_RW(est_sat=est_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=target)


@lru_cache(maxsize=8)
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


def _make_static_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - TimeConstants.sec2cent,
        R=7000.0 * np.array([0.0, -np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]),
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
        case "stop_rotation":
            return No_Goal()
        case "align_x":
            return ECI_Goal(np.array([1.0, 0.0, 0.0]))
        case "align_diag":
            return ECI_Goal(normalize(np.array([1.0, 1.0, 1.0])))
        case "desaturate":
            return ECI_Goal(np.array([1.0, 0.0, 0.0]))
        case "full_task":
            return ECI_Goal(normalize(np.array([1.0, 1.0, 1.0])))
        case _:
            raise ValueError(f"Unknown scenario '{name}'.")


def _scenario_state(name: str) -> State:
    rng_state = np.random.get_state()
    np.random.seed(1)
    try:
        if name == "stop_rotation":
            w0 = random_n_unit_vec(3) * np.random.uniform(1.0, 2.0) * np.pi / 180.0
            q0 = normalize(random_n_unit_vec(4))
            h0 = np.zeros(3)
        elif name == "align_x":
            w0 = np.zeros(3)
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
            h0 = np.ones(3)
        elif name == "align_diag":
            w0 = np.zeros(3)
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
            h0 = np.ones(3)
        elif name == "desaturate":
            w0 = np.zeros(3)
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
            h0 = np.array([0.5, 0.0, 0.0])
        elif name == "full_task":
            w0 = np.zeros(3)
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
            h0 = np.full(3, 0.5)
        else:
            raise ValueError(f"Unknown scenario '{name}'.")
        return State(w=w0, q=normalize(q0), h=h0)
    finally:
        np.random.set_state(rng_state)


def _scenario_cfg(name: str) -> dict:
    return STOP_CFG if name == "stop_rotation" else FULL_CFG if name in {"desaturate", "full_task"} else POINTING_CFG


def run_mtq_w_rw_simulation(
    scenario_name: str,
    *,
    tf: float | None = None,
    dt: float | None = None,
    real_orbit: bool | None = None,
    progress: bool = False,
) -> MTQwRWRun:
    default_tf, default_dt, default_real_orbit = SCENARIO_DEFAULTS[scenario_name]
    horizon = default_tf if tf is None else tf
    step = default_dt if dt is None else dt
    use_real_orbit = default_real_orbit if real_orbit is None else real_orbit

    x = _scenario_state(scenario_name)
    satellite = _make_satellite(rw_h=x.h)
    cfg = _scenario_cfg(scenario_name)
    controller = _make_controller(satellite, **cfg)
    goal = _scenario_goal(scenario_name)
    orbit = _make_real_orbit(horizon, step) if use_real_orbit else _make_static_orbit(horizon, step)

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
        x = State.from_array(out.y[:, -1])
        x = x.normalized()
        t = next_t

    return MTQwRWRun(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        sensor_hist=sensor_hist,
        u_hist=u_hist,
        boresight_hist=boresight_hist,
        goal=goal,
    )


def _expected_command(
    controller: MTQ_w_RW,
    satellite: Satellite,
    x_hat: State,
    sens: np.ndarray,
    os_hat: Orbital_State,
    goal: Goal,
) -> np.ndarray:
    w = x_hat.w
    q = x_hat.q
    sens_clean = np.asarray(sens, dtype=float).copy()
    sens_clean[np.isnan(sens_clean)] = 0.0
    b_body = np.asarray(controller.M_mtm_read @ sens_clean, float).reshape(3,)

    boresight = satellite.get_boresight(goal.boresight_name)
    q_err = goal.error(q=q, body_boresight=boresight, os0=os_hat)
    _, w_ref_eci = goal.to_ref(os0=os_hat)
    w_ref_body = rot_mat(q).T @ w_ref_eci
    tau_att = -controller.p_gain * q_err - controller.d_gain * (w - w_ref_body)

    rw_axes = np.vstack(
        [
            np.asarray(actuator.axis, dtype=float).reshape(3,)
            for actuator in satellite.actuators
            if isinstance(actuator, RW)
        ]
    )
    h_vals = x_hat.h
    h_rw_body = h_vals @ rw_axes
    tau_att = tau_att + np.cross(w, satellite.J_0 @ w + h_rw_body)

    tau_dump = -controller.c_gain * (h_rw_body - controller.h_target)
    m_mag_eff = -skewsym(b_body) @ controller.A_mtq
    u_mtq = np.linalg.pinv(m_mag_eff) @ tau_dump
    u_mtq = limit(u_mtq, controller.max_torque)

    tau_mag_actual = m_mag_eff @ u_mtq
    tau_rw_req = tau_att - tau_mag_actual
    u_rw = controller.M_rw_act @ tau_rw_req
    u_rw = limit(u_rw, controller.max_torque)
    return u_mtq + u_rw


def _final_rate_norm(run: MTQwRWRun) -> float:
    return float(np.linalg.norm(run.state_hist[-1, 0:3]))


def _final_average_command(run: MTQwRWRun) -> float:
    return float(np.max(np.mean(np.abs(run.u_hist[-5:, :]), axis=0)))


def _final_alignment_error_deg(run: MTQwRWRun, target_vec_eci: np.ndarray) -> float:
    q_err_vec = vector_alignment_error(
        q=run.state_hist[-1, 3:7],
        eci_goal=target_vec_eci,
        body_boresight=np.array([0.0, 0.0, 1.0]),
    )
    return float(np.degrees(np.linalg.norm(q_err_vec)))


@pytest.fixture
def mtq_rw_satellite() -> Satellite:
    return _make_satellite()


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
def base_state() -> State:
    q = normalize(np.array([0.8, 0.1, -0.2, 0.55]))
    w = np.array([0.03, -0.02, 0.01])
    h = np.array([0.5, -0.2, 0.1])
    return State(w=w, q=q, h=h)


def test_mtq_w_rw_rejects_unachievable_target_momentum(mtq_rw_satellite: Satellite) -> None:
    with pytest.raises(ValueError):
        _make_controller(
            mtq_rw_satellite,
            p_gain=0.1,
            d_gain=0.7,
            c_gain=0.1,
            h_target=np.array([4.0, 0.0, 0.0]),
        )


def test_mtq_w_rw_zero_state_zero_goal_returns_zero_command(
    mtq_rw_satellite: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(mtq_rw_satellite, p_gain=0.0, d_gain=0.0, c_gain=0.0)
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(3))
    sens = mtq_rw_satellite.sensor_readings(x=state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=mtq_rw_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )

    np.testing.assert_allclose(command, np.zeros(len(mtq_rw_satellite.actuators)))


def test_mtq_w_rw_cleans_nan_sensor_values(
    mtq_rw_satellite: Satellite,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(mtq_rw_satellite, **POINTING_CFG)
    goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    sens = mtq_rw_satellite.sensor_readings(x=base_state, os=base_orbital_state)
    sens_with_nan = sens.copy()
    sens_with_nan[1] = np.nan

    command = controller.find_u(
        x_hat=base_state,
        sens=sens_with_nan,
        est_sat=mtq_rw_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = _expected_command(
        controller,
        mtq_rw_satellite,
        base_state,
        sens_with_nan,
        base_orbital_state,
        goal,
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_w_rw_matches_pointing_and_dump_allocation_math(
    mtq_rw_satellite: Satellite,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(
        mtq_rw_satellite,
        p_gain=0.1,
        d_gain=0.7,
        c_gain=0.1,
        h_target=np.array([0.0, 0.0, 0.0]),
    )
    goal = StaticGoal(q_err=np.array([0.2, -0.1, 0.05]), w_ref_eci=np.array([0.01, 0.0, -0.02]))
    sens = mtq_rw_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=base_state,
        sens=sens,
        est_sat=mtq_rw_satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )
    expected = _expected_command(
        controller,
        mtq_rw_satellite,
        base_state,
        sens,
        base_orbital_state,
        goal,
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_w_rw_no_goal_matches_rate_damping_and_dump_math(
    mtq_rw_satellite: Satellite,
    base_state: np.ndarray,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(
        mtq_rw_satellite,
        p_gain=0.0,
        d_gain=1.0,
        c_gain=0.1,
        h_target=np.zeros(3),
    )
    sens = mtq_rw_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=base_state,
        sens=sens,
        est_sat=mtq_rw_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    expected = _expected_command(
        controller,
        mtq_rw_satellite,
        base_state,
        sens,
        base_orbital_state,
        No_Goal(),
    )

    np.testing.assert_allclose(command, expected)


def test_mtq_w_rw_includes_gyroscopic_compensation(
    mtq_rw_satellite: Satellite,
    base_orbital_state: Orbital_State,
) -> None:
    controller = _make_controller(mtq_rw_satellite, p_gain=0.0, d_gain=0.0, c_gain=0.0)
    state = State(w=[0.04, -0.03, 0.02], q=[1.0, 0.0, 0.0, 0.0], h=[0.3, -0.2, 0.1])
    sens = mtq_rw_satellite.sensor_readings(x=state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=mtq_rw_satellite,
        os_hat=base_orbital_state,
        goal=No_Goal(),
    )
    expected = _expected_command(
        controller,
        mtq_rw_satellite,
        state,
        sens,
        base_orbital_state,
        No_Goal(),
    )

    np.testing.assert_allclose(command, expected)
    assert np.any(np.abs(command[3:]) > 0.0)


def test_mtq_w_rw_respects_mtq_and_rw_limits(
    base_orbital_state: Orbital_State,
) -> None:
    satellite = _make_satellite()
    controller = _make_controller(
        satellite,
        p_gain=10.0,
        d_gain=10.0,
        c_gain=10.0,
        h_target=np.array([-3.0, 3.0, -3.0]),
    )
    state = State(w=[5.0, -4.0, 3.0], q=[1.0, 0.0, 0.0, 0.0], h=[2.0, -2.0, 2.0])
    goal = StaticGoal(q_err=np.array([1.0, -1.0, 0.5]), w_ref_eci=np.array([1.0, -1.0, 0.5]))
    sens = satellite.sensor_readings(x=state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=state,
        sens=sens,
        est_sat=satellite,
        os_hat=base_orbital_state,
        goal=goal,
    )

    np.testing.assert_array_less(np.abs(command[:3]), controller.max_torque[:3] + 1.0e-12)
    np.testing.assert_array_less(np.abs(command[3:]), controller.max_torque[3:] + 1.0e-12)



@pytest.mark.parametrize(
    ("scenario_name", "alignment_target", "alignment_tol_deg", "settle_tol", "rate_tol", "momentum_index", "momentum_tol"),
    [
        ("stop_rotation", None, None, 1.0e-4, 1.0e-3, None, None),
        ("align_diag", normalize(np.array([1.0, 1.0, 1.0])), 1.0, 1.0e-3, 1.0e-3, None, None),
        ("desaturate", None, None, None, None, 0, 0.1),
        ("full_task", normalize(np.array([1.0, 1.0, 1.0])), 1.0, None, 1.0e-3, None, None),
    ],
)
def test_mtq_w_rw_converges_in_representative_scenarios(
    scenario_name: str,
    alignment_target: np.ndarray | None,
    alignment_tol_deg: float | None,
    settle_tol: float | None,
    rate_tol: float | None,
    momentum_index: int | None,
    momentum_tol: float | None,
) -> None:
    run = run_mtq_w_rw_simulation(scenario_name)

    if rate_tol is not None:
        assert _final_rate_norm(run) < rate_tol
    if settle_tol is not None:
        assert _final_average_command(run) < settle_tol
    if alignment_target is not None and alignment_tol_deg is not None:
        assert _final_alignment_error_deg(run, alignment_target) < alignment_tol_deg
    if momentum_index is not None and momentum_tol is not None:
        assert abs(run.state_hist[-1, 7 + momentum_index]) < momentum_tol


def debug_plots(run: MTQwRWRun) -> None:
    from ADCS.helpers.plotting.animate_estimator import animate_attitude
    from ADCS.helpers.plotting.animate_orbit import animate_orbit
    from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
    from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum
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
    animate_orbit(
        time=run.time_hist,
        state_hist=run.state_hist,
        os_hist=run.os_hist,
        boresight_goal_hist=run.boresight_hist,
    )
    create_close_all_button_window()


def main() -> None:
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "stop_rotation"
    if scenario_name not in SCENARIO_DEFAULTS:
        raise SystemExit(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIO_DEFAULTS)}")

    run = run_mtq_w_rw_simulation(scenario_name, progress=True)
    debug_plots(run)


if __name__ == "__main__":
    main()
