"""Compiled three-axis magnetometer measurements and attitude information."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    attitude_vector_jacobian_angle,
    attitude_vector_jacobian_mrp,
    attitude_vector_jacobian_quaternion,
    rotate_inertial_to_sensor,
)
from GenADCS.backends.python.helpers.random.philox import next_draw

from .kernels import magnetometer_information, vector_measurement
from .sensor import Sensor


@njit(cache=True)
def _saturate(value, limit):
    return np.minimum(np.maximum(value, -limit), limit)


class Magnetometer:
    """Body-to-sensor mounting with componentwise saturation and Gaussian noise."""

    def __init__(
        self, position, orientation, misalignment, saturation_value, noise, bias
    ):
        base = Sensor(
            position, orientation, misalignment, saturation_value, noise, bias
        )
        self.position = base.position
        self.orientation = base.orientation
        self.misalignment = base.misalignment
        self.saturation_value = base.saturation_value
        self.noise = base.noise
        self.noise_cholesky = np.linalg.cholesky(self.noise)
        self.bias = base.bias
        self.enabled = base.enabled

    def noiseless_biasfree_measurement(self, field_inertial, attitude):
        return vector_measurement(
            field_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            False,
            False,
        )

    def noiseless_biased_measurement(self, field_inertial, attitude):
        return vector_measurement(
            field_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            True,
            False,
        )

    def noisy_biasfree_measurement(self, field_inertial, attitude):
        return vector_measurement(
            field_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            False,
            True,
            next_draw(self, 3),
        )

    def noisy_biased_measurement(self, field_inertial, attitude):
        return vector_measurement(
            field_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            True,
            True,
            next_draw(self, 3),
        )

    def jacobian_bias(self):
        return np.eye(3)

    def jacobian_relative_quaternion(self, field_inertial, attitude, delta_quaternion):
        field = rotate_inertial_to_sensor(
            field_inertial, attitude, self.orientation, self.misalignment
        )
        return attitude_vector_jacobian_quaternion(field, delta_quaternion)

    def jacobian_relative_mrp(self, field_inertial, attitude, delta_mrp):
        field = rotate_inertial_to_sensor(
            field_inertial, attitude, self.orientation, self.misalignment
        )
        return attitude_vector_jacobian_mrp(field, delta_mrp)

    def jacobian_relative_angle_vector(self, field_inertial, attitude, delta_angle):
        field = rotate_inertial_to_sensor(
            field_inertial, attitude, self.orientation, self.misalignment
        )
        return attitude_vector_jacobian_angle(field, delta_angle)

    def availability(self):
        return self.enabled

    def fisher_information_matrix(self, field_inertial, attitude, field_covariance):
        if not self.enabled:
            return np.zeros((3, 3))
        return magnetometer_information(
            field_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.noise,
            field_covariance,
        )
