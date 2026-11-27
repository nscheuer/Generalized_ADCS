"""Attitude reference goals backed by Rust."""

from .._subsystem import (
    AlongTrackGoal,
    AngularRateGoal,
    CoordinateFMCGoal,
    CoordinateGoal,
    CrossTrackGoal,
    Goal,
    NadirGoal,
    NoGoal,
    QuaternionGoal,
    SunGoal,
    TwoVectorGoal,
    VectorGoal,
    VelocityGoal,
    ZenithGoal,
)

nogoal = NoGoal
quaterniongoal = QuaternionGoal
vectorgoal = VectorGoal
twovectorgoal = TwoVectorGoal
angularrategoal = AngularRateGoal
nadirgoal = NadirGoal
zenithgoal = ZenithGoal
alongtrackgoal = AlongTrackGoal
crosstrackgoal = CrossTrackGoal
velocitygoal = VelocityGoal
coordinategoal = CoordinateGoal
sungoal = SunGoal
coordinatefmcgoal = CoordinateFMCGoal

__all__ = [
    "Goal",
    "NoGoal", "QuaternionGoal", "VectorGoal", "TwoVectorGoal", "AngularRateGoal",
    "NadirGoal", "ZenithGoal", "AlongTrackGoal", "CrossTrackGoal", "VelocityGoal",
    "CoordinateGoal", "SunGoal", "CoordinateFMCGoal",
    "nogoal", "quaterniongoal", "vectorgoal", "twovectorgoal", "angularrategoal",
    "nadirgoal", "zenithgoal", "alongtrackgoal", "crosstrackgoal", "velocitygoal",
    "coordinategoal", "sungoal", "coordinatefmcgoal",
]
