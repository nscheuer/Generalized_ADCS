"""Numba-compatible fine Sun sensor returning two boresight angles."""

import numpy as np

from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    fine_sun_angles,
    fine_sun_attitude_jacobian,
    fine_sun_fisher_information,
)
from GenADCS.backends.python.helpers.random.philox import next_draw

from .kernels import fine_direction_available, fine_direction_measurement


class FineSunSensor:
    """Two-axis angular Sun sensor; angles and FOV limits are in radians."""

    def __init__(
        self, position, orientation, misalignment, noise, bias, fov_half_angles
    ):
        if (
            position.shape != (3,)
            or orientation.shape != (4,)
            or misalignment.shape != (4,)
        ):
            raise ValueError("position must be a 3-vector and orientations quaternions")
        if noise.shape != (2, 2) or bias.shape != (2,) or fov_half_angles.shape != (2,):
            raise ValueError("noise must be 2x2; bias and FOV must be 2-vectors")
        self.position = position
        self.orientation = orientation / np.sqrt(np.dot(orientation, orientation))
        self.misalignment = misalignment / np.sqrt(np.dot(misalignment, misalignment))
        self.noise = noise
        self.noise_cholesky = np.linalg.cholesky(noise)
        self.bias = bias
        self.fov_half_angles = fov_half_angles
        self.enabled = True

    def noiseless_biasfree_measurement(self, sun_inertial, attitude):
        return fine_direction_measurement(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            False,
            False,
        )

    def noiseless_biased_measurement(self, sun_inertial, attitude):
        return fine_direction_measurement(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            True,
            False,
        )

    def noisy_biasfree_measurement(self, sun_inertial, attitude):
        return fine_direction_measurement(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            False,
            True,
            next_draw(self, 2),
        )

    def noisy_biased_measurement(self, sun_inertial, attitude):
        return fine_direction_measurement(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            True,
            True,
            next_draw(self, 2),
        )

    def jacobian_bias(self):
        return np.eye(2)

    def jacobian_relative_quaternion(self, sun_inertial, attitude, delta_quaternion):
        return fine_sun_attitude_jacobian(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            delta_quaternion,
            0,
        )

    def jacobian_relative_mrp(self, sun_inertial, attitude, delta_mrp):
        return fine_sun_attitude_jacobian(
            sun_inertial, attitude, self.orientation, self.misalignment, delta_mrp, 1
        )

    def jacobian_relative_angle_vector(self, sun_inertial, attitude, delta_angle):
        return fine_sun_attitude_jacobian(
            sun_inertial, attitude, self.orientation, self.misalignment, delta_angle, 2
        )

    def availability(self, sun_inertial, attitude, sun_visible=True):
        return fine_direction_available(
            sun_inertial,
            attitude,
            self.orientation,
            self.misalignment,
            self.fov_half_angles,
            self.enabled,
            sun_visible,
        )

    def fisher_information_matrix(self, sun_inertial, attitude, sun_visible=True):
        if not self.availability(sun_inertial, attitude, sun_visible):
            return np.zeros((3, 3))
        angles = fine_sun_angles(
            sun_inertial, attitude, self.orientation, self.misalignment
        )
        return fine_sun_fisher_information(
            angles, self.noise, self.orientation, self.misalignment
        )
