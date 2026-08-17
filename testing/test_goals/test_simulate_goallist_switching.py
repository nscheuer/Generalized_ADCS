import numpy as np

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.simulate import simulate
from ADCS.state import State


UNIT_VECTORS = MathConstants.unitvecs


def run_goallist_switch_simulation():
    start_time = 0.22
    dt = 1.0
    final_time = 60.0
    switch_time = start_time + 30.0 * TimeConstants.sec2cent

    vector_a = normalize(np.array([1.0, 0.0, 0.0]))
    vector_b = normalize(np.array([0.0, 1.0, 0.0]))
    goals = GoalList(
        {start_time: ECI_Goal(vector_a.copy()), switch_time: ECI_Goal(vector_b.copy())},
        time_units="centuries",
    )

    satellite = Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=UNIT_VECTORS[index], max_torque=0.1) for index in range(3)],
        sensors=[MTM(axis=UNIT_VECTORS[index]) for index in range(3)],
    )
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=start_time,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    result = simulate(x=state, satellite=satellite, goal=goals, os0=orbital_state, dt=dt, tf=final_time)[0]
    target_history = np.asarray(result.target_hist, dtype=float)
    directions = target_history[:, 1:4]
    normalized_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return normalized_directions, vector_a, vector_b, start_time, switch_time, dt


def test_simulate_goallist_switch_logs_vector_mode_targets():
    target_history, _, _, _, _, _ = run_goallist_switch_simulation()
    assert target_history.ndim == 2
    assert target_history.shape[1] == 3
    assert target_history.shape[0] > 10


def test_simulate_goallist_switch_uses_only_goal_a_or_goal_b_directions():
    directions, vector_a, vector_b, _, _, _ = run_goallist_switch_simulation()
    matches_a = np.all(np.isclose(directions, vector_a, atol=1e-9), axis=1)
    matches_b = np.all(np.isclose(directions, vector_b, atol=1e-9), axis=1)
    assert np.all(matches_a | matches_b)


def test_simulate_goallist_switch_starts_with_goal_a_and_ends_with_goal_b():
    directions, vector_a, vector_b, _, _, _ = run_goallist_switch_simulation()
    matches_a = np.all(np.isclose(directions, vector_a, atol=1e-9), axis=1)
    matches_b = np.all(np.isclose(directions, vector_b, atol=1e-9), axis=1)
    assert matches_a[0]
    assert matches_b[-1]


def test_simulate_goallist_switch_has_exactly_one_transition():
    directions, vector_a, _, _, _, _ = run_goallist_switch_simulation()
    matches_a = np.all(np.isclose(directions, vector_a, atol=1e-9), axis=1)
    transitions = int(np.sum(np.diff(matches_a.astype(int)) != 0))
    assert transitions == 1


def test_simulate_goallist_switch_happens_at_scheduled_time_within_one_step():
    directions, vector_a, _, start_time, switch_time, dt = run_goallist_switch_simulation()
    matches_a = np.all(np.isclose(directions, vector_a, atol=1e-9), axis=1)
    switch_index = int(np.argmax(~matches_a))
    actual_switch_time = start_time + switch_index * dt * TimeConstants.sec2cent
    assert abs(actual_switch_time - switch_time) <= 1.5 * dt * TimeConstants.sec2cent
