"""Nadir pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _nadir


class NadirGoal(OrbitGoal):
    """Point body +Z toward the central body; +X follows orbit tangent."""

    kernel = staticmethod(_nadir)


nadirgoal = NadirGoal
