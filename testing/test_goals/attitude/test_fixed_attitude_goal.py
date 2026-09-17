import numpy as np

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.helpers.math_helpers import normalize

from testing.test_goals._helpers import make_test_orbital_state


def test_fixed_attitude_goal_normalizes_input_quaternion():
    goal = Fixed_Attitude_Goal(np.array([2.0, 0.0, 0.0, 0.0]))

    assert np.allclose(goal.q_ref, np.array([1.0, 0.0, 0.0, 0.0]))


def test_fixed_attitude_goal_to_ref_returns_constant_reference():
    q_ref = normalize(np.array([0.4, -0.1, 0.2, 0.8]))
    goal = Fixed_Attitude_Goal(q_ref)

    r_ref, w_ref = goal.to_ref(make_test_orbital_state(R=[7100.0, 5.0, -2.0]))

    assert np.allclose(r_ref, q_ref)
    assert np.allclose(w_ref, np.zeros(3))
