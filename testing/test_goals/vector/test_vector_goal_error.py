import numpy as np

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.helpers.math_helpers import normalize

from testing.test_goals._helpers import make_test_orbital_state


class ConstantVectorGoal(Vector_Goal):
    def __init__(self, vector):
        super().__init__()
        self.vector = np.asarray(vector, dtype=float)

    def to_ref(self, os0):
        r_ref = np.empty(4)
        r_ref[0] = np.nan
        r_ref[1:] = normalize(self.vector)
        return r_ref, np.zeros(3)


def test_vector_goal_error_is_zero_for_perfect_alignment():
    goal = ConstantVectorGoal([0.0, 0.0, 1.0])

    error = goal.error(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    assert np.allclose(error, np.zeros(3), atol=1e-12)


def test_vector_goal_error_matches_quarter_turn_geometry():
    goal = ConstantVectorGoal([1.0, 0.0, 0.0])

    error = goal.error(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    expected = np.array([0.0, -1.0 / np.sqrt(2.0), 0.0])
    assert np.allclose(error, expected)


def test_vector_goal_error_handles_180_degree_case_with_unit_axis():
    goal = ConstantVectorGoal([0.0, 0.0, -1.0])

    error = goal.error(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    assert np.isclose(np.linalg.norm(error), 1.0)
    assert np.isclose(np.dot(error, np.array([0.0, 0.0, 1.0])), 0.0)


def test_vector_goal_error_uses_body_to_eci_quaternion():
    goal = ConstantVectorGoal([1.0, 0.0, 0.0])
    q_body_to_eci = normalize(np.array([1.0, 0.0, 1.0, 0.0]))

    error = goal.error(
        q=q_body_to_eci,
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    assert np.allclose(error, np.zeros(3), atol=1e-12)
