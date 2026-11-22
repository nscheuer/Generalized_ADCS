from GenADCS.backends.cpp.dynamics._dynamics import static_gravity_gradient_torque
"""Central-field gravity-gradient torque for a fixed inertia tensor."""

import numpy as np
from numba import njit

from GenADCS.backends.cpp.helpers.math.frames.frames import inertial_to_body



class StaticGravityGradientModel:
    def __init__(self, geometry):
        self.inertia_b = np.ascontiguousarray(geometry.inertia, dtype=np.float64)
        if self.inertia_b.shape != (3, 3) or not np.all(np.isfinite(self.inertia_b)):
            raise ValueError("geometry.inertia must be a finite 3x3 matrix")
        if not np.allclose(self.inertia_b, self.inertia_b.T):
            raise ValueError("geometry.inertia must be symmetric")

    def evaluate(self, position_i, q_b_from_i, environment):
        return static_gravity_gradient_torque(
            np.asarray(position_i, dtype=np.float64),
            np.asarray(q_b_from_i, dtype=np.float64), self.inertia_b,
            float(environment.planetary_body.gravitational_parameter),
        )
