import numpy as np
import pytest

from ADCS.CONOPS.goals import (
    AntiBField_Goal,
    AntiSun_Goal,
    BField_Goal,
    ECI_Goal,
    PerpBField_Goal,
    Sun_Goal,
)

from testing.test_goals._helpers import make_test_orbital_state


def test_eci_goal_normalizes_input_and_has_zero_rate():
    goal = ECI_Goal(np.array([2.0, 0.0, 0.0]))

    r_ref, w_ref = goal.to_ref(make_test_orbital_state())

    assert np.isnan(r_ref[0])
    assert np.allclose(r_ref[1:], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(w_ref, np.zeros(3))


@pytest.mark.parametrize(
    ("goal_cls", "direction_getter", "sign"),
    [
        (Sun_Goal, lambda os0: os0.get_sun_eci(), 1.0),
        (AntiSun_Goal, lambda os0: os0.get_sun_eci(), -1.0),
        (BField_Goal, lambda os0: os0.get_b_eci(), 1.0),
        (AntiBField_Goal, lambda os0: os0.get_b_eci(), -1.0),
    ],
)
def test_environment_vector_goals_return_expected_direction(goal_cls, direction_getter, sign):
    goal = goal_cls()
    os0 = make_test_orbital_state()
    expected_direction = direction_getter(os0)
    expected_direction = sign * expected_direction / np.linalg.norm(expected_direction)

    r_ref, w_ref = goal.to_ref(os0)

    assert np.isnan(r_ref[0])
    assert np.allclose(r_ref[1:], expected_direction)
    assert np.allclose(w_ref, np.zeros(3))


def test_perp_bfield_goal_is_perpendicular_to_bfield_and_uses_orbital_rate():
    goal = PerpBField_Goal()
    os0 = make_test_orbital_state(B=[0.0, 0.0, 2.0e-5], V=[0.0, 7.5, 0.0])

    r_ref, w_ref = goal.to_ref(os0)

    assert np.isclose(np.dot(r_ref[1:], os0.B / np.linalg.norm(os0.B)), 0.0)
    assert np.isclose(np.linalg.norm(r_ref[1:]), 1.0)
    assert np.allclose(w_ref, np.cross(os0.R, os0.V) / np.dot(os0.R, os0.R))


def test_perp_bfield_goal_uses_fallback_when_bfield_parallel_to_velocity():
    goal = PerpBField_Goal()
    os0 = make_test_orbital_state(B=[2.0e-5, 0.0, 0.0], V=[7.5, 0.0, 0.0])

    r_ref, _ = goal.to_ref(os0)

    assert np.isclose(np.dot(r_ref[1:], np.array([1.0, 0.0, 0.0])), 0.0)
    assert np.isclose(np.linalg.norm(r_ref[1:]), 1.0)
