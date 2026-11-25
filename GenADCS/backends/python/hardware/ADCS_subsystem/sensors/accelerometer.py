"""Numba-compatible three-axis specific-force accelerometer."""

import numpy as np
from GenADCS.backends.python.helpers.random.philox import next_draw

from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    rotate_inertial_to_sensor,
    attitude_vector_jacobian_quaternion,
    attitude_vector_jacobian_mrp,
    attitude_vector_jacobian_angle,
)
from GenADCS.backends.python.helpers.math.vector import saturate_vector
from .sensor import Sensor
from .kernels import vector_measurement


class Accelerometer:
    """Measures inertial specific force, rotated into sensor/body axes.

    Inputs ``acceleration_inertial`` and ``gravity_inertial`` are combined as
    ``a - g`` (specific force), then transformed using the shared mounting.
    """

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

    def noiseless_biasfree_measurement(
        self, acceleration_inertial, gravity_inertial, attitude
    ):
        return vector_measurement(
            acceleration_inertial - gravity_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            False,
            False,
        )

    def noiseless_biased_measurement(
        self, acceleration_inertial, gravity_inertial, attitude
    ):
        return vector_measurement(
            acceleration_inertial - gravity_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            True,
            False,
        )

    def noisy_biasfree_measurement(
        self, acceleration_inertial, gravity_inertial, attitude
    ):
        return vector_measurement(
            acceleration_inertial - gravity_inertial,
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

    def noisy_biased_measurement(
        self, acceleration_inertial, gravity_inertial, attitude
    ):
        return vector_measurement(
            acceleration_inertial - gravity_inertial,
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

    def jacobian_relative_quaternion(
        self, acceleration_inertial, gravity_inertial, attitude, delta_quaternion
    ):
        force = rotate_inertial_to_sensor(
            acceleration_inertial - gravity_inertial,
            attitude,
            self.orientation,
            self.misalignment,
        )
        return attitude_vector_jacobian_quaternion(force, delta_quaternion)

    def jacobian_relative_mrp(
        self, acceleration_inertial, gravity_inertial, attitude, delta_mrp
    ):
        force = rotate_inertial_to_sensor(
            acceleration_inertial - gravity_inertial,
            attitude,
            self.orientation,
            self.misalignment,
        )
        return attitude_vector_jacobian_mrp(force, delta_mrp)

    def jacobian_relative_angle_vector(
        self, acceleration_inertial, gravity_inertial, attitude, delta_angle
    ):
        force = rotate_inertial_to_sensor(
            acceleration_inertial - gravity_inertial,
            attitude,
            self.orientation,
            self.misalignment,
        )
        return attitude_vector_jacobian_angle(force, delta_angle)

    def jacobian_bias(self):
        return np.eye(3)

    def availability(self):
        return self.enabled
