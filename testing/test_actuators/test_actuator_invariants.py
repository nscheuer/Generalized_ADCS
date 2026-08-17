import numpy as np
import pytest

from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import Actuator, MTQ, RW
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.state import State


def make_orbital_state(j2000: float = 0.22) -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=j2000,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=1e-5 * np.array([1.0, 2.0, -3.0]) / np.linalg.norm([1.0, 2.0, -3.0]),
    )


def make_state() -> State:
    return State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])


def make_biased_rw(axis: np.ndarray) -> RW:
    return RW(
        axis=axis,
        max_torque=10.0,
        J=0.02,
        h=0.0,
        h_max=5.0,
        bias=Bias(bias=0.3, std_bias=1.0),
        noise=Noise(noise=0.0, std_noise=1.0),
    )


def test_base_actuator_torque_is_zero_vector():
    actuator = Actuator(axis=np.array([1.0, 0.0, 0.0]), u_max=1.0)
    torque = np.asarray(actuator.torque(u=0.5, x=make_state(), os=make_orbital_state()))
    assert torque.shape == (3,)
    assert torque.dtype.kind == "f"
    assert np.array_equal(torque, np.zeros(3))


def test_mtq_torque_is_linear_below_saturation():
    np.random.seed(12345)
    mtq = MTQ(axis=random_n_unit_vec(3), max_torque=2.0)
    orbital_state = make_orbital_state()
    state = make_state()
    torque_at = lambda command: mtq.torque(u=command, x=state, os=orbital_state)
    assert np.allclose(torque_at(0.5), torque_at(0.0) + 0.25 * torque_at(2.0))


def test_mtq_positive_saturation_clamps_to_limit():
    np.random.seed(12345)
    mtq = MTQ(axis=random_n_unit_vec(3), max_torque=2.0)
    orbital_state = make_orbital_state()
    state = make_state()
    limited = mtq.torque(u=2.0, x=state, os=orbital_state)
    with pytest.warns(UserWarning):
        saturated = mtq.torque(u=20.0, x=state, os=orbital_state)
    assert np.allclose(saturated, limited)


def test_mtq_negative_saturation_clamps_to_limit():
    np.random.seed(12345)
    mtq = MTQ(axis=random_n_unit_vec(3), max_torque=2.0)
    orbital_state = make_orbital_state()
    state = make_state()
    positive_limit = mtq.torque(u=2.0, x=state, os=orbital_state)
    negative_limit = mtq.torque(u=-2.0, x=state, os=orbital_state)
    with pytest.warns(UserWarning):
        saturated = mtq.torque(u=-20.0, x=state, os=orbital_state)
    assert np.allclose(saturated, negative_limit)
    assert np.allclose(saturated, -positive_limit)


def test_rw_torque_matches_command_below_saturation():
    np.random.seed(2024)
    rw = RW(axis=random_n_unit_vec(3), max_torque=0.05, J=0.01, h=0.0, h_max=0.2)
    torque = rw.torque(u=0.025, x=make_state(), os=make_orbital_state())
    assert np.allclose(torque, rw.axis * 0.025)


def test_rw_storage_torque_matches_command_below_saturation():
    np.random.seed(2024)
    rw = RW(axis=random_n_unit_vec(3), max_torque=0.05, J=0.01, h=0.0, h_max=0.2)
    storage_torque = rw.storage_torque(u=0.025, x=make_state(), os=make_orbital_state())
    assert np.isclose(storage_torque, -0.025)


def test_rw_positive_saturation_clamps_torque_and_storage():
    np.random.seed(2024)
    rw = RW(axis=random_n_unit_vec(3), max_torque=0.05, J=0.01, h=0.0, h_max=0.2)
    with pytest.warns(UserWarning):
        torque = rw.torque(u=0.365, x=make_state(), os=make_orbital_state())
    with pytest.warns(UserWarning):
        storage_torque = rw.storage_torque(u=0.365, x=make_state(), os=make_orbital_state())
    assert np.allclose(torque, rw.axis * 0.05)
    assert np.isclose(storage_torque, -0.05)


def test_rw_negative_saturation_clamps_torque_and_storage():
    np.random.seed(2024)
    rw = RW(axis=random_n_unit_vec(3), max_torque=0.05, J=0.01, h=0.0, h_max=0.2)
    with pytest.warns(UserWarning):
        torque = rw.torque(u=-0.455, x=make_state(), os=make_orbital_state())
    with pytest.warns(UserWarning):
        storage_torque = rw.storage_torque(u=-0.455, x=make_state(), os=make_orbital_state())
    assert np.allclose(torque, -rw.axis * 0.05)
    assert np.isclose(storage_torque, 0.05)


def test_rw_update_momentum_clamps_positive_scalar():
    rw = RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0, J=0.01, h=0.0, h_max=0.1)
    with pytest.warns(UserWarning):
        rw.update_momentum(0.5)
    assert np.isclose(rw.h, 0.1)


def test_rw_update_momentum_clamps_negative_scalar():
    rw = RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0, J=0.01, h=0.0, h_max=0.1)
    with pytest.warns(UserWarning):
        rw.update_momentum(-0.5)
    assert np.isclose(rw.h, -0.1)


def test_rw_update_momentum_keeps_in_range_scalar_values():
    rw = RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0, J=0.01, h=0.0, h_max=0.1)
    rw.update_momentum(0.07)
    assert np.isclose(rw.h, 0.07)
    rw.update_momentum(-0.07)
    assert np.isclose(rw.h, -0.07)


def test_rw_update_momentum_clamps_vector_magnitude():
    rw = RW(
        axis=np.array([0.0, 0.0, 1.0]),
        max_torque=1.0,
        J=0.01,
        h=np.zeros(3),
        h_max=np.array([0.1, 0.1, 0.1]),
    )
    target = np.array([0.3, -0.4, 0.0])
    with pytest.warns(UserWarning):
        rw.update_momentum(target)
    momentum = np.asarray(rw.h)
    assert momentum.shape == (3,)
    assert np.isclose(np.linalg.norm(momentum), 0.1)
    assert np.allclose(momentum / np.linalg.norm(momentum), target / np.linalg.norm(target))


def test_rw_update_momentum_accepts_small_vector():
    rw = RW(
        axis=np.array([0.0, 0.0, 1.0]),
        max_torque=1.0,
        J=0.01,
        h=np.zeros(3),
        h_max=np.array([0.1, 0.1, 0.1]),
    )
    target = np.array([0.01, 0.02, 0.0])
    rw.update_momentum(target)
    assert np.allclose(np.asarray(rw.h), target)


def test_rw_body_and_storage_torques_cancel_with_noise_and_bias():
    np.random.seed(98765)
    rw = make_biased_rw(random_n_unit_vec(3))
    orbital_state = make_orbital_state()
    state = State(w=0.01 * np.array([1.0, 0.0, 0.0]), q=[1.0, 0.0, 0.0, 0.0])

    residuals = []
    for _ in range(200):
        orbital_state.J2000 += TimeConstants.sec2cent
        body_torque = rw.torque(u=0.4, x=state, os=orbital_state)
        storage_torque = rw.storage_torque(u=0.4, x=state, os=orbital_state)
        residuals.append(float(np.dot(body_torque, rw.axis) + storage_torque))

    assert np.max(np.abs(residuals)) < 1e-12


def test_rw_reaction_is_call_order_independent():
    np.random.seed(13)
    rw = make_biased_rw(random_n_unit_vec(3))
    orbital_state = make_orbital_state()
    state = make_state()

    for step in range(50):
        orbital_state.J2000 += TimeConstants.sec2cent
        if step % 2 == 0:
            first_torque = rw.torque(u=0.4, x=state, os=orbital_state)
            storage_torque = rw.storage_torque(u=0.4, x=state, os=orbital_state)
            repeated_torque = rw.torque(u=0.4, x=state, os=orbital_state)
        else:
            storage_torque = rw.storage_torque(u=0.4, x=state, os=orbital_state)
            first_torque = rw.torque(u=0.4, x=state, os=orbital_state)
            repeated_torque = rw.torque(u=0.4, x=state, os=orbital_state)

        assert np.array_equal(first_torque, repeated_torque)
        assert abs(float(np.dot(first_torque, rw.axis)) + storage_torque) < 1e-12


def test_rw_dynamics_step_conserves_total_angular_momentum():
    np.random.seed(555)
    rw = make_biased_rw(random_n_unit_vec(3))
    orbital_state = make_orbital_state()
    state = make_state()
    dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

    leaks = []
    for _ in range(50):
        orbital_state.J2000 += TimeConstants.sec2cent
        body_torque = rw.torque(u=0.4, x=state, os=orbital_state, dmode=dmode)
        storage_torque = rw.storage_torque(u=0.4, x=state, os=orbital_state, dmode=dmode)
        leaks.append(float(np.dot(body_torque, rw.axis) + storage_torque))

    assert np.max(np.abs(leaks)) < 1e-12
