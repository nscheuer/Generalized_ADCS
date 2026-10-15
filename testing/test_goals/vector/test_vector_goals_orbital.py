import numpy as np
import pytest

from ADCS.CONOPS.goals import (
    AntiVelocity_Goal,
    LVLH_Tangential_Goal,
    Nadir_Goal,
    Velocity_Goal,
    Zenith_Goal,
)

from testing.test_goals._helpers import make_test_orbital_state


@pytest.mark.parametrize(
    ("goal_cls", "expected_direction"),
    [
        (Nadir_Goal, np.array([-1.0, 0.0, 0.0])),
        (Zenith_Goal, np.array([1.0, 0.0, 0.0])),
        (Velocity_Goal, np.array([0.0, 1.0, 0.0])),
        (AntiVelocity_Goal, np.array([0.0, -1.0, 0.0])),
        (LVLH_Tangential_Goal, np.array([0.0, 1.0, 0.0])),
    ],
)
def test_orbital_vector_goals_return_expected_direction(goal_cls, expected_direction):
    goal = goal_cls()

    r_ref, _ = goal.to_ref(make_test_orbital_state())

    assert np.isnan(r_ref[0])
    assert np.allclose(r_ref[1:], expected_direction)


@pytest.mark.parametrize(
    "goal_cls",
    [Nadir_Goal, Zenith_Goal, Velocity_Goal, AntiVelocity_Goal, LVLH_Tangential_Goal],
)
def test_orbital_vector_goals_return_orbital_rate(goal_cls):
    goal = goal_cls()
    os0 = make_test_orbital_state()

    _, w_ref = goal.to_ref(os0)

    expected = np.cross(os0.R, os0.V) / np.dot(os0.R, os0.R)
    assert np.allclose(w_ref, expected)
