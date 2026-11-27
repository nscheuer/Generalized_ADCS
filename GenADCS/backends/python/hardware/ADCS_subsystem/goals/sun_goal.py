"""Sun-pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _sun


class SunGoal(OrbitGoal):
    """Point body +Z toward the Sun using the configured DE421 ephemeris."""

    kernel = staticmethod(_sun)


sungoal = SunGoal
