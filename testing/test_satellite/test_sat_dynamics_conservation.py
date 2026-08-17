import numpy as np
import pytest

from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.state import State


@pytest.fixture(scope="module")
def orbital_state():
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.zeros(3),
        S=np.array([1e8, 0.0, 0.0]),
        rho=0.0,
    )


def body_inertia() -> np.ndarray:
    return np.diagflat([0.5, 0.8, 1.2])


def inertial_angular_momentum(satellite: Satellite, state: State, axes: np.ndarray | None = None) -> np.ndarray:
    body_momentum = satellite.J_COM @ state.w
    if axes is not None and state.h.size:
        body_momentum = body_momentum + axes.T @ state.h
    return rot_mat(state.q) @ body_momentum


def rotational_kinetic_energy(satellite: Satellite, state: State) -> float:
    return 0.5 * state.w @ (satellite.J_COM @ state.w)


COM_OFFSETS = [
    np.array([0.05, 0.02, -0.03]),
    np.array([0.20, 0.0, 0.0]),
    np.array([0.0, -0.15, 0.0]),
    np.array([0.10, 0.10, -0.10]),
    np.array([1e-6, 0.0, 0.0]),
]
COM_IDS = ["small", "x_axis", "y_axis", "multi", "near_zero"]


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_torque_free_motion_conserves_angular_momentum_with_com_offset(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    state = State(w=[0.02, -0.015, 0.01], q=[1.0, 0.0, 0.0, 0.0])
    initial_momentum = inertial_angular_momentum(satellite, state)

    max_drift = 0.0
    for _ in range(4000):
        state = satellite.noiseless_rk4(state, np.zeros(0), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(
            max_drift,
            np.linalg.norm(inertial_angular_momentum(satellite, state) - initial_momentum) / np.linalg.norm(initial_momentum),
        )

    assert max_drift < 1e-6


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_torque_free_motion_conserves_rotational_energy_with_com_offset(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    state = State(w=[0.02, -0.015, 0.01], q=[1.0, 0.0, 0.0, 0.0])
    initial_energy = rotational_kinetic_energy(satellite, state)

    max_drift = 0.0
    for _ in range(4000):
        state = satellite.noiseless_rk4(state, np.zeros(0), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(max_drift, abs(rotational_kinetic_energy(satellite, state) - initial_energy) / abs(initial_energy))

    assert max_drift < 1e-6


def test_torque_free_motion_with_rw_conserves_total_angular_momentum(orbital_state):
    from ADCS.helpers.math_helpers import normalize
    from ADCS.satellite_hardware.actuators import RW

    wheel = RW(axis=normalize(np.array([1.0, 0.6, 0.3])), max_torque=1.0, J=0.05, h=0.02, h_max=10.0)
    satellite = Satellite(mass=3.0, COM=np.array([0.04, -0.02, 0.05]), J_0=body_inertia(), actuators=[wheel])
    axes = np.vstack([actuator.axis for actuator in satellite.rw_actuators])
    state = State(w=[0.02, -0.015, 0.01], q=[1.0, 0.0, 0.0, 0.0], h=[0.02])
    initial_momentum = inertial_angular_momentum(satellite, state, axes)

    max_drift = 0.0
    for _ in range(3000):
        state = satellite.noiseless_rk4(state, np.zeros(1), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(
            max_drift,
            np.linalg.norm(inertial_angular_momentum(satellite, state, axes) - initial_momentum) / np.linalg.norm(initial_momentum),
        )

    assert max_drift < 1e-5


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_dynamics_core_matches_analytic_euler_equation(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    rng = np.random.default_rng(0)
    for _ in range(20):
        omega = rng.normal(size=3) * 0.05
        state = State(w=omega, q=[1.0, 0.0, 0.0, 0.0])
        state_dot = satellite.dynamics_core(x=state, u=np.zeros(0), orbital_state=orbital_state)
        analytic = satellite.invJ_COM @ (-np.cross(omega, satellite.J_COM @ omega))
        assert np.allclose(state_dot[:3], analytic, atol=1e-10)


def test_dynjac_matches_finite_difference_with_com_offset(orbital_state):
    from ADCS.satellite_hardware.actuators import MTQ

    satellite = Satellite(
        mass=4.0,
        COM=np.array([0.05, 0.02, -0.03]),
        J_0=body_inertia(),
        actuators=[MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)],
    )
    state = State(w=[0.03, -0.02, 0.012], q=[1.0, 0.0, 0.0, 0.0])
    control = np.zeros(1)

    def dynamics(candidate):
        return satellite.dynamics_core(x=candidate, u=control, orbital_state=orbital_state)

    jacobian = satellite.dynJacCore(state, control, orbital_state)[0]
    state_array = state.as_array()
    numeric = np.zeros((state_array.size, state_array.size))
    for index in range(state_array.size):
        delta = np.zeros(state_array.size)
        delta[index] = 1e-7
        numeric[:, index] = (dynamics(State.from_array(state_array + delta)) - dynamics(State.from_array(state_array - delta))) / (2e-7)

    assert np.allclose(jacobian.T[: state_array.size, : state_array.size], numeric, atol=1e-4)


def test_noiseless_rk4_does_not_mutate_input_state(orbital_state):
    satellite = Satellite(J_0=body_inertia())
    state = State(w=[0.01, -0.008, 0.006], q=[2.0, 0.0, 0.0, 0.0])
    before = state.copy()
    satellite.noiseless_rk4(state, np.zeros(0), 0.5, orbital_state, orbital_state, mid_orbital_state=orbital_state)
    assert np.array_equal(state.as_array(), before.as_array())
