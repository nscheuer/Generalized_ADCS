from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ADCS.controller import BDot
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_helpers import limit
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State


@dataclass(frozen=True)
class BDotRun:
    time_hist: np.ndarray
    state_hist: np.ndarray
    os_hist: list[Orbital_State]
    sensor_hist: np.ndarray
    u_hist: np.ndarray


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _make_satellite(*, include_rw: bool = False, mtq_max_torque: float = 0.1) -> Satellite:
    mtqs = [MTQ(axis=axis, max_torque=mtq_max_torque) for axis in MathConstants.unitvecs]
    actuators = list(mtqs)

    if include_rw:
        rws = [
            RW(axis=axis, max_torque=4.51, J=0.22, h=0.0, h_max=3.8)
            for axis in MathConstants.unitvecs
        ]
        actuators.extend(rws)

    mtms = [MTM(axis=axis) for axis in MathConstants.unitvecs]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=actuators,
        sensors=mtms,
    )


def _make_static_orbit(tf: float, dt: float, *, b_eci: np.ndarray) -> Orbit:
    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8.0, 0.0, 0.0]),
        B=b_eci,
        S=np.array([1e5 + 1, 0.0, 0.0]),
        rho=5e-12,
    )
    steps = int(tf / dt) + 20
    orbs = []
    for step in range(steps):
        current = os0.copy()
        current.J2000 = os0.J2000 + step * dt * TimeConstants.sec2cent
        orbs.append(current)
    return Orbit(orbs)


def _make_real_orbit(tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time_j2000 = 0.22
    start_date = start_time_j2000 - TimeConstants.sec2cent
    end_date = start_time_j2000 + tf * TimeConstants.sec2cent
    os0 = Orbital_State(
        ephem=ephem,
        J2000=start_date,
        R=7000.0 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8.0, 0.0, 0.0]),
    )
    return Orbit(os0=os0, end_time=end_date, dt=dt, zonal_J=2, fast=False)


def run_bdot_simulation(
    *,
    tf: float = 500.0,
    dt: float = 1.0,
    real_orbit: bool = False,
    include_rw: bool = True,
    progress: bool = False,
) -> BDotRun:
    rng = np.random.default_rng(1)
    satellite = _make_satellite(
        include_rw=include_rw,
        mtq_max_torque=0.01 if include_rw else 0.1,
    )

    w0 = _unit(rng.normal(size=3)) * rng.uniform(1.0, 2.0) * np.pi / 180.0
    q0 = _unit(rng.normal(size=4))
    x = State(w=w0, q=q0, h=np.zeros(3) if include_rw else np.empty(0))
    state_dim = 10 if include_rw else 7

    orbit = (
        _make_real_orbit(tf, dt)
        if real_orbit
        else _make_static_orbit(tf, dt, b_eci=np.array([0.0, 0.1, 0.0]))
    )
    controller = BDot(est_sat=satellite, gain=100.0)

    steps = int(tf / dt)
    time_hist = np.full(steps, np.nan)
    state_hist = np.full((steps, state_dim), np.nan)
    sensor_hist = np.full((steps, len(satellite.sensors + satellite.rw_actuators)), np.nan)
    u_hist = np.full((steps, len(satellite.actuators)), np.nan)
    os_hist: list[Orbital_State] = []

    iterator = tqdm(range(steps), desc=f"Simulating BDot (RW={include_rw})") if progress else range(steps)
    t = 0.0
    start_time_j2000 = 0.22

    for ind in iterator:
        current_j2000 = start_time_j2000 + t * TimeConstants.sec2cent
        os = orbit.get_os(J2000=current_j2000)
        sens = satellite.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=satellite, os_hat=os)

        time_hist[ind] = t
        state_hist[ind, :] = x.as_array()
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        os_hist.append(os)

        next_t = t + dt
        prev_os = os.copy()
        next_os = orbit.get_os(start_time_j2000 + next_t * TimeConstants.sec2cent)
        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0.0, dt),
            y0=x.as_array(),
            method="RK45",
            args=(u, prev_os, next_os),
            rtol=1e-7,
            atol=1e-7,
        )
        x = State.from_array(out.y[:, -1]).normalized()
        t = next_t

    return BDotRun(
        time_hist=time_hist,
        state_hist=state_hist,
        os_hist=os_hist,
        sensor_hist=sensor_hist,
        u_hist=u_hist,
    )


@pytest.fixture
def bdot_satellite() -> Satellite:
    return _make_satellite(include_rw=False, mtq_max_torque=0.1)


@pytest.fixture
def bdot_satellite_with_rw() -> Satellite:
    return _make_satellite(include_rw=True, mtq_max_torque=0.01)


@pytest.fixture
def base_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([1.0e-5, -2.0e-5, 3.0e-5]),
    )


@pytest.fixture
def base_state() -> State:
    q = _unit(np.array([0.8, 0.2, -0.3, 0.4]))
    w = np.array([0.01, -0.02, 0.03])
    return State(w=w, q=q)


def test_bdot_first_call_returns_zero_command(
    bdot_satellite: Satellite,
    base_state: State,
    base_orbital_state: Orbital_State,
) -> None:
    controller = BDot(est_sat=bdot_satellite, gain=100.0)
    sens = bdot_satellite.sensor_readings(x=base_state, os=base_orbital_state)

    command = controller.find_u(
        x_hat=base_state,
        sens=sens,
        est_sat=bdot_satellite,
        os_hat=base_orbital_state,
    )

    np.testing.assert_allclose(command, np.zeros(3))


def test_bdot_second_call_matches_finite_difference_law(
    bdot_satellite: Satellite,
    base_state: State,
    base_orbital_state: Orbital_State,
) -> None:
    controller = BDot(est_sat=bdot_satellite, gain=25.0)
    os0 = base_orbital_state.copy()
    os1 = base_orbital_state.copy()
    os1.B = np.array([1.6e-5, -1.4e-5, 2.1e-5])
    os1.J2000 = os0.J2000 + 2.0 * TimeConstants.sec2cent

    sens0 = bdot_satellite.sensor_readings(x=base_state, os=os0)
    sens1 = bdot_satellite.sensor_readings(x=base_state, os=os1)

    controller.find_u(x_hat=base_state, sens=sens0, est_sat=bdot_satellite, os_hat=os0)
    command = controller.find_u(x_hat=base_state, sens=sens1, est_sat=bdot_satellite, os_hat=os1)

    b0 = controller.M_read @ sens0
    b1 = controller.M_read @ sens1
    bdot = (b1 - b0) / 2.0
    expected = limit(-(controller.gain * bdot), controller.max_torque)

    np.testing.assert_allclose(command, expected)


def test_bdot_zero_time_step_returns_zero_derivative_command(
    bdot_satellite: Satellite,
    base_state: State,
    base_orbital_state: Orbital_State,
) -> None:
    controller = BDot(est_sat=bdot_satellite, gain=100.0)
    os0 = base_orbital_state.copy()
    os1 = base_orbital_state.copy()
    os1.B = np.array([3.0e-5, 1.0e-5, -2.0e-5])
    os1.J2000 = os0.J2000

    sens0 = bdot_satellite.sensor_readings(x=base_state, os=os0)
    sens1 = bdot_satellite.sensor_readings(x=base_state, os=os1)

    controller.find_u(x_hat=base_state, sens=sens0, est_sat=bdot_satellite, os_hat=os0)
    command = controller.find_u(x_hat=base_state, sens=sens1, est_sat=bdot_satellite, os_hat=os1)

    np.testing.assert_allclose(command, np.zeros(3))


def test_bdot_command_is_saturated_to_mtq_limits(bdot_satellite: Satellite) -> None:
    controller = BDot(est_sat=bdot_satellite, gain=1e9)
    sens0 = np.array([0.0, 0.0, 0.0])
    sens1 = np.array([1.0, -2.0, 3.0])
    os0 = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([0.0, 0.0, 0.0]),
    )
    os1 = os0.copy()
    os1.J2000 = os0.J2000 + 1.0 * TimeConstants.sec2cent

    controller.find_u(x_hat=np.zeros(7), sens=sens0, est_sat=bdot_satellite, os_hat=os0)
    command = controller.find_u(x_hat=np.zeros(7), sens=sens1, est_sat=bdot_satellite, os_hat=os1)

    np.testing.assert_allclose(command, np.array([-0.1, 0.1, -0.1]))


def test_bdot_with_reaction_wheels_only_commands_mtqs(
    bdot_satellite_with_rw: Satellite,
    base_state: State,
    base_orbital_state: Orbital_State,
) -> None:
    controller = BDot(est_sat=bdot_satellite_with_rw, gain=50.0)
    state_with_rw = State(w=base_state.w, q=base_state.q, h=np.zeros(3))
    os0 = base_orbital_state.copy()
    os1 = base_orbital_state.copy()
    os1.B = np.array([-2.0e-5, 1.0e-5, 2.5e-5])
    os1.J2000 = os0.J2000 + 1.0 * TimeConstants.sec2cent

    sens0 = bdot_satellite_with_rw.sensor_readings(x=state_with_rw, os=os0)
    sens1 = bdot_satellite_with_rw.sensor_readings(x=state_with_rw, os=os1)

    controller.find_u(x_hat=state_with_rw, sens=sens0, est_sat=bdot_satellite_with_rw, os_hat=os0)
    command = controller.find_u(x_hat=state_with_rw, sens=sens1, est_sat=bdot_satellite_with_rw, os_hat=os1)

    assert command.shape == (6,)
    np.testing.assert_allclose(command[3:], np.zeros(3))


def test_bdot_reconstructs_body_field_from_mtm_measurements(
    bdot_satellite: Satellite,
    base_state: State,
    base_orbital_state: Orbital_State,
) -> None:
    controller = BDot(est_sat=bdot_satellite, gain=10.0)
    sens = bdot_satellite.sensor_readings(x=base_state, os=base_orbital_state)
    expected_body_field = base_orbital_state.get_state_vector(x=base_state)["b"]

    np.testing.assert_allclose(controller.M_read @ sens, expected_body_field)



def test_bdot_full_convergence_loop() -> None:
    results = run_bdot_simulation(
        tf=500.0,
        dt=1.0,
        real_orbit=False,
        include_rw=True,
        progress=False,
    )

    final_u = np.mean(np.abs(results.u_hist[-10:, 0:3]), axis=0)
    np.testing.assert_array_less(final_u, 1e-3)

    final_w = results.state_hist[-1, 0:3]
    q_final = results.state_hist[-1, 3:7]
    final_b_body = results.os_hist[-1].get_state_vector(x=State.from_array(results.state_hist[-1]))["b"]
    b_unit = final_b_body / np.linalg.norm(final_b_body)
    w_perp = final_w - np.dot(final_w, b_unit) * b_unit

    assert np.linalg.norm(w_perp) < 1e-2
    assert np.linalg.norm(q_final) == pytest.approx(1.0, abs=1e-6)


def debug_plots(results: BDotRun) -> None:
    from ADCS.helpers.plotting.animate_estimator import animate_attitude
    from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
    from ADCS.helpers.plotting.plot_controller import plot_control
    from ADCS.helpers.plotting.plot_estimator import plot_state_comparison

    animate_attitude(time=results.time_hist, state_hist=results.state_hist, os_hist=results.os_hist)
    plot_control(time=results.time_hist, u_hist=results.u_hist)
    plot_state_comparison(time=results.time_hist, state_hist=results.state_hist)
    create_close_all_button_window()


def main() -> None:
    results = run_bdot_simulation(
        tf=100.0,
        dt=1.0,
        real_orbit=False,
        include_rw=True,
        progress=True,
    )
    debug_plots(results)


if __name__ == "__main__":
    main()
