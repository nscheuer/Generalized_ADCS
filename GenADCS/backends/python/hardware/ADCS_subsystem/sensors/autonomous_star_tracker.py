"""Numba-compatible autonomous star-tracker measurement model."""

import numpy as np

from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    _conjugate, _multiply, _normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import _quaternion_to_dcm
from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    star_tracker_angle_jacobian, small_rotation_noise_quaternion,
)
from .sensor import Sensor
from .kernels import (star_tracker_measurement, star_tracker_residual,
                      star_tracker_mrp_jacobian, star_tracker_information)


class AutonomousStarTracker:
    """Attitude sensor. Noise covariance is in the tracker frame."""

    def __init__(self, position, orientation, misalignment, saturation_value, noise, bias):
        base = Sensor(position, orientation, misalignment, saturation_value, noise, bias)
        self.position = base.position
        self.orientation = base.orientation
        self.misalignment = base.misalignment
        self.saturation_value = base.saturation_value
        self.noise = base.noise
        self.noise_cholesky = np.linalg.cholesky(self.noise)
        self.bias = base.bias
        self.enabled = base.enabled

    def noiseless_biasfree_measurement(self, attitude):
        return star_tracker_measurement(attitude, self.orientation, self.misalignment,
                                        self.noise_cholesky, False)

    def noiseless_biased_measurement(self, attitude):
        return self.noiseless_biasfree_measurement(attitude)

    def noisy_biasfree_measurement(self, attitude):
        return star_tracker_measurement(attitude, self.orientation, self.misalignment,
                                        self.noise_cholesky, True)

    def noisy_biased_measurement(self, attitude):
        return self.noisy_biasfree_measurement(attitude)

    def residual(self, measured_attitude, estimated_attitude):
        return star_tracker_residual(measured_attitude, estimated_attitude)

    def jacobian_relative_quaternion(self):
        return 2.0 * np.eye(3)

    def jacobian_relative_mrp(self, delta_mrp):
        return star_tracker_mrp_jacobian(delta_mrp)

    def jacobian_relative_angle_vector(self, delta_angle):
        return star_tracker_angle_jacobian(delta_angle)

    def availability(self):
        return self.enabled

    def fisher_information_matrix(self):
        if not self.enabled:
            return np.zeros((3, 3))
        # Mounting maps body attitude error into tracker coordinates.
        return star_tracker_information(self.noise, self.orientation, self.misalignment)
