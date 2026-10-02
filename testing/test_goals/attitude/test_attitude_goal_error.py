import numpy as np

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.helpers.math_helpers import normalize, quat_inv, quat_mult

from testing.test_goals._helpers import make_test_orbital_state


def test_attitude_goal_error_is_zero_at_reference_attitude():
    q_ref = normalize(np.array([0.9, 0.1, -0.2, 0.3]))
    goal = Fixed_Attitude_Goal(q_ref)

    error = goal.error(
        q=q_ref,
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    assert np.allclose(error, np.zeros(3), atol=1e-12)


def test_attitude_goal_error_matches_known_quaternion_difference():
    q_ref = normalize(np.array([1.0, 1.0, 0.0, 0.0]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    goal = Fixed_Attitude_Goal(q_ref)

    error = goal.error(
        q=q,
        body_boresight=np.array([1.0, 0.0, 0.0]),
        os0=make_test_orbital_state(),
    )

    q_err = quat_mult(quat_inv(q_ref), q)
    if q_err[0] < 0.0:
        q_err = -q_err

    assert np.allclose(error, q_err[1:4])


def test_attitude_goal_error_is_invariant_to_reference_sign():
    q_ref = normalize(np.array([0.7, -0.2, 0.1, 0.6]))
    q = normalize(np.array([0.8, 0.1, -0.3, 0.5]))

    goal_pos = Fixed_Attitude_Goal(q_ref)
    goal_neg = Fixed_Attitude_Goal(-q_ref)

    error_pos = goal_pos.error(q=q, body_boresight=np.zeros(3), os0=make_test_orbital_state())
    error_neg = goal_neg.error(q=q, body_boresight=np.zeros(3), os0=make_test_orbital_state())

    assert np.allclose(error_pos, error_neg)
