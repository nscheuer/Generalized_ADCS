"""No-goal reference with no attitude or rate target."""

from numba import njit

from .goal import Goal, validate_reference_state


@njit(cache=True)
def _no_goal_reference(state, epoch_jd, runtime):
    return None, None


class NoGoal(Goal):
    """Return no references, indicating that attitude control is inactive."""

    kernel = staticmethod(_no_goal_reference)

    def reference(self, state, epoch_jd, runtime):
        value = validate_reference_state(state)
        return self.kernel(value, float(epoch_jd), runtime)


nogoal = NoGoal
