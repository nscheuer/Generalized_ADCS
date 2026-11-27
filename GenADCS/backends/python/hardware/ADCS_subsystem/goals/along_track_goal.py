"""Along-track pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _alongtrack


class AlongTrackGoal(OrbitGoal):
    """Point body +Z along the orbit transverse (along-track) direction."""

    kernel = staticmethod(_alongtrack)


alongtrackgoal = AlongTrackGoal
