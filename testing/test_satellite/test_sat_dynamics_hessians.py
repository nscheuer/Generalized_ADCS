import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.satellite.satellite import Satellite


@pytest.fixture(scope="module")
def hessian_case():
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.zeros(3),
        S=np.array([1e8, 0.0, 0.0]),
        rho=0.0,
    )
    satellite = Satellite(
        mass=4.0,
        J_0=np.diagflat([0.5, 0.8, 1.2]),
        actuators=[MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)],
    )
    state = np.hstack(([0.02, -0.01, 0.015], [1.0, 0.0, 0.0, 0.0]))
    control = np.zeros(1)
    return satellite, state, control, orbital_state


def finite_difference_dynamics_hessian(satellite: Satellite, state: np.ndarray, control: np.ndarray, orbital_state: Orbital_State):
    analytic = np.asarray(satellite.dynamics_Hessians(state, control, orbital_state)[0][0], dtype=float)
    numeric = np.zeros((state.size, state.size, analytic.shape[-1]))
    for index in range(state.size):
        delta = np.zeros(state.size)
        delta[index] = 1e-6
        plus = np.asarray(satellite.dynJacCore(state + delta, control, orbital_state)[0], dtype=float)
        minus = np.asarray(satellite.dynJacCore(state - delta, control, orbital_state)[0], dtype=float)
        numeric[index] = (plus - minus) / (2.0e-6)
    return analytic, numeric


@pytest.mark.slow
def test_dynamics_hessian_has_expected_state_shape(hessian_case):
    satellite, state, control, orbital_state = hessian_case
    analytic = np.asarray(satellite.dynamics_Hessians(state, control, orbital_state)[0][0], dtype=float)
    assert analytic.shape[0] >= state.size
    assert analytic.shape[1] >= state.size


@pytest.mark.slow
def test_dynamics_hessian_matches_finite_difference_of_jacobian(hessian_case):
    analytic, numeric = finite_difference_dynamics_hessian(*hessian_case)
    error = np.max(np.abs(analytic[: numeric.shape[0], : numeric.shape[1], :] - numeric))
    assert error < 1e-4
