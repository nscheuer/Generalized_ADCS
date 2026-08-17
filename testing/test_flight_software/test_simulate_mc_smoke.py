import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.mc.simulate_mc import simulate_mc
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State

UNIT_VECTORS = MathConstants.unitvecs


@pytest.fixture(scope="module")
def mc_result():
    satellite = Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=UNIT_VECTORS[index], max_torque=0.1) for index in range(3)],
        sensors=[MTM(axis=UNIT_VECTORS[index]) for index in range(3)],
    )
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    return simulate_mc(x=state, satellite=satellite, os0=orbital_state, dt=1.0, tf=8.0, num_runs=2, max_workers=1, base_seed=0), satellite


def test_simulate_mc_returns_expected_number_of_runs(mc_result):
    result, _ = mc_result
    assert len(result.runs) == 2


def test_simulate_mc_each_run_has_finite_state_history(mc_result):
    result, satellite = mc_result
    for run in result.runs:
        state_history = State.stack(run.state_hist)
        assert state_history.ndim == 2
        assert state_history.shape[0] >= 5
        assert state_history.shape[1] == satellite.state_len
        assert np.all(np.isfinite(state_history))


def test_simulate_mc_each_run_keeps_quaternions_normalized(mc_result):
    result, _ = mc_result
    for run in result.runs:
        quaternion_norms = np.linalg.norm(State.stack(run.state_hist)[:, 3:7], axis=1)
        np.testing.assert_allclose(quaternion_norms, 1.0, atol=1e-3)


def test_simulate_mc_aggregate_indexing_and_iteration_work(mc_result):
    result, _ = mc_result
    assert result[0] is result.runs[0]
    assert len(list(iter(result))) == 2
