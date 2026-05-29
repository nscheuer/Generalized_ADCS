import numpy as np
import pytest

from ADCS.CONOPS.goals import Attitude_Goal, Goal

from testing.test_goals._helpers import make_test_orbital_state


def test_goal_base_to_ref_raises():
    goal = Goal()

    with pytest.raises(NotImplementedError):
        goal.to_ref(make_test_orbital_state())


def test_goal_base_error_raises():
    goal = Goal()

    with pytest.raises(NotImplementedError):
        goal.error(
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            body_boresight=np.array([0.0, 0.0, 1.0]),
            os0=make_test_orbital_state(),
        )


def test_attitude_goal_base_to_ref_raises():
    goal = Attitude_Goal()

    with pytest.raises(NotImplementedError):
        goal.to_ref(make_test_orbital_state())
