import numpy as np

from ADCS.CONOPS.goals import No_Goal

from testing.test_goals._helpers import make_test_orbital_state


def test_no_goal_returns_zero_reference_and_rate():
    goal = No_Goal()

    r_ref, w_ref = goal.to_ref(make_test_orbital_state())

    assert np.array_equal(r_ref, np.zeros(4))
    assert np.array_equal(w_ref, np.zeros(3))


def test_no_goal_returns_zero_error():
    goal = No_Goal()

    error = goal.error(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        body_boresight=np.array([0.0, 0.0, 1.0]),
        os0=make_test_orbital_state(),
    )

    assert np.array_equal(error, np.zeros(3))
