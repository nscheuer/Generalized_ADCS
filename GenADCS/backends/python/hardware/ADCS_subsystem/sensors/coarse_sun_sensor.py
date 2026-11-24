"""Numba-compatible photodiode coarse Sun sensor."""

import numpy as np
from GenADCS.backends.python.helpers.random.philox import next_draw

from GenADCS.backends.python.helpers.math.rotations.sensor_kernels import (
    sun_in_body, coarse_sun_measurement, coarse_sun_attitude_jacobian,
    mounting_dcm,
)
from .kernels import (scalar_sun_measurement, coarse_sun_available,
                      coarse_sun_information)


class CoarseSunSensor:
    """One photodiode; mounting quaternions map body axes to sensor axes.

    ``normal`` is expressed in sensor coordinates. ``noise_variance`` is the
    scalar measurement variance, and ``fov_half_angle`` is in radians.
    """

    def __init__(self, position, orientation, misalignment, normal,
                 sensitivity, saturation_value, noise_variance, bias,
                 fov_half_angle):
        if position.shape != (3,) or normal.shape != (3,):
            raise ValueError("position and normal must have 3 components")
        if orientation.shape != (4,) or misalignment.shape != (4,):
            raise ValueError("orientation and misalignment must be quaternions")
        norm = np.sqrt(np.dot(normal, normal))
        if norm == 0.0 or sensitivity <= 0.0 or saturation_value <= 0.0:
            raise ValueError("normal must be nonzero; sensitivity and saturation positive")
        if noise_variance <= 0.0 or fov_half_angle <= 0.0 or fov_half_angle > np.pi / 2.0:
            raise ValueError("noise variance and FOV must be positive; FOV <= pi/2")
        self.position = position
        self.orientation = orientation / np.sqrt(np.dot(orientation, orientation))
        self.misalignment = misalignment / np.sqrt(np.dot(misalignment, misalignment))
        self.normal = normal / norm
        self.sensitivity = sensitivity
        self.saturation_value = saturation_value
        self.noise_variance = noise_variance
        self.noise_std = np.sqrt(noise_variance)
        self.bias = bias
        self.fov_half_angle = fov_half_angle
        self.enabled = True

    def noiseless_biasfree_measurement(self, sun_inertial, attitude):
        return scalar_sun_measurement(sun_inertial, attitude, self.orientation, self.misalignment,
                                      self.normal, self.sensitivity, self.bias, self.noise_std,
                                      self.saturation_value, False, False)

    def noiseless_biased_measurement(self, sun_inertial, attitude):
        return scalar_sun_measurement(sun_inertial, attitude, self.orientation, self.misalignment,
                                      self.normal, self.sensitivity, self.bias, self.noise_std,
                                      self.saturation_value, True, False)

    def noisy_biasfree_measurement(self, sun_inertial, attitude):
        return scalar_sun_measurement(sun_inertial, attitude, self.orientation, self.misalignment,
                                      self.normal, self.sensitivity, self.bias, self.noise_std,
                                      self.saturation_value, False, True, next_draw(self, 1))

    def noisy_biased_measurement(self, sun_inertial, attitude):
        return scalar_sun_measurement(sun_inertial, attitude, self.orientation, self.misalignment,
                                      self.normal, self.sensitivity, self.bias, self.noise_std,
                                      self.saturation_value, True, True, next_draw(self, 1))

    def jacobian_bias(self):
        return 1.0

    def jacobian_relative_quaternion(self, sun_inertial, attitude, delta_quaternion):
        return coarse_sun_attitude_jacobian(sun_inertial, attitude, self.orientation,
                    self.misalignment, self.normal, self.sensitivity, delta_quaternion, 0)

    def jacobian_relative_mrp(self, sun_inertial, attitude, delta_mrp):
        return coarse_sun_attitude_jacobian(sun_inertial, attitude, self.orientation,
                    self.misalignment, self.normal, self.sensitivity, delta_mrp, 1)

    def jacobian_relative_angle_vector(self, sun_inertial, attitude, delta_angle):
        return coarse_sun_attitude_jacobian(sun_inertial, attitude, self.orientation,
                    self.misalignment, self.normal, self.sensitivity, delta_angle, 2)

    def availability(self, sun_inertial, attitude, sun_visible=True):
        return coarse_sun_available(sun_inertial, attitude, self.orientation,
                                    self.misalignment, self.normal, self.fov_half_angle,
                                    self.enabled, sun_visible)

    def fisher_information_matrix(self, sun_inertial, attitude, sun_visible=True):
        if not self.availability(sun_inertial, attitude, sun_visible):
            return np.zeros((3, 3))
        return coarse_sun_information(sun_inertial, attitude, self.orientation,
                                      self.misalignment, self.normal, self.sensitivity,
                                      self.noise_variance)
