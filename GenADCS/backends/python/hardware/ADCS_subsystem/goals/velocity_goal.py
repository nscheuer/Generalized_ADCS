"""Velocity-direction pointing reference goal."""

from ._orbit_goal import OrbitGoal
from ._orbital_kernels import _velocity


class VelocityGoal(OrbitGoal):
    """Point body +Z along instantaneous inertial velocity."""

    kernel = staticmethod(_velocity)


velocitygoal = VelocityGoal
