"""Cross-track pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _crosstrack


class CrossTrackGoal(OrbitGoal):
    """Point body +Z along the orbit normal; +X follows the radial direction."""

    kernel = staticmethod(_crosstrack)


crosstrackgoal = CrossTrackGoal
