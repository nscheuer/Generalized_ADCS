"""Attitude reference goals for ADCS controllers.

Goal objects validate and hold configuration; their numerical reference
kernels are Numba compiled. Each ``reference`` call returns the desired
body-from-inertial quaternion and body-frame angular rate, with ``None`` for
any component the goal does not specify.
"""

from .angular_rate_goal import AngularRateGoal, angularrategoal
from .along_track_goal import AlongTrackGoal, alongtrackgoal
from .coordinate_fmc_goal import CoordinateFMCGoal, coordinatefmcgoal
from .coordinate_goal import CoordinateGoal
from .cross_track_goal import CrossTrackGoal, crosstrackgoal
from .goal import Goal
from .nadir_goal import NadirGoal, nadirgoal
from .no_goal import NoGoal, nogoal
from .quaternion_goal import QuaternionGoal, quaterniongoal
from .sun_goal import SunGoal, sungoal
from .two_vector_goal import TwoVectorGoal, twovectorgoal
from .velocity_goal import VelocityGoal, velocitygoal
from .vector_goal import VectorGoal, vectorgoal
from .zenith_goal import ZenithGoal, zenithgoal

coordinategoal = CoordinateGoal

__all__ = [
    "Goal",
    "NoGoal", "QuaternionGoal", "VectorGoal", "TwoVectorGoal", "AngularRateGoal",
    "NadirGoal", "ZenithGoal", "AlongTrackGoal", "CrossTrackGoal", "VelocityGoal",
    "CoordinateGoal", "SunGoal", "CoordinateFMCGoal",
    "nogoal", "quaterniongoal", "vectorgoal", "twovectorgoal", "angularrategoal",
    "nadirgoal", "zenithgoal", "alongtrackgoal", "crosstrackgoal", "velocitygoal",
    "coordinategoal", "sungoal", "coordinatefmcgoal",
]
