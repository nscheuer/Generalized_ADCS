"""Zenith pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _zenith


class ZenithGoal(OrbitGoal):
    """Point body +Z away from the central body; +X follows orbit tangent."""

    kernel = staticmethod(_zenith)


zenithgoal = ZenithGoal
