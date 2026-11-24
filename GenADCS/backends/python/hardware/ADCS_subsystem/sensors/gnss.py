"""Numba-compatible inertial position and velocity GNSS measurement model."""

import numpy as np
from .kernels import gnss_measurement


class GNSS:
    """Six-axis position/velocity sensor with inertial-frame bias and noise.

    ``noise`` is the 6x6 position/velocity covariance. Bias order is position
    (m), then velocity (m/s). Measurement output follows the same ordering.
    """

    def __init__(self, noise, bias):
        if noise.shape != (6, 6) or bias.shape != (6,):
            raise ValueError("noise must be 6x6 and bias must have 6 components")
        self.noise = noise
        self.noise_cholesky = np.linalg.cholesky(noise)
        self.bias = bias
        self.enabled = True

    def noiseless_biasfree_measurement(self, position_inertial, velocity_inertial):
        return gnss_measurement(position_inertial, velocity_inertial, self.bias,
                                self.noise_cholesky, False, False)

    def noiseless_biased_measurement(self, position_inertial, velocity_inertial):
        return gnss_measurement(position_inertial, velocity_inertial, self.bias,
                                self.noise_cholesky, True, False)

    def noisy_biasfree_measurement(self, position_inertial, velocity_inertial):
        return gnss_measurement(position_inertial, velocity_inertial, self.bias,
                                self.noise_cholesky, False, True)

    def noisy_biased_measurement(self, position_inertial, velocity_inertial):
        return gnss_measurement(position_inertial, velocity_inertial, self.bias,
                                self.noise_cholesky, True, True)

    def jacobian_relative_quaternion(self):
        return np.zeros((6, 3))

    def jacobian_relative_mrp(self):
        return np.zeros((6, 3))

    def jacobian_relative_angle_vector(self):
        return np.zeros((6, 3))

    def jacobian_position(self):
        result = np.zeros((6, 3))
        result[0:3, 0:3] = np.eye(3)
        return result

    def jacobian_velocity(self):
        result = np.zeros((6, 3))
        result[3:6, 0:3] = np.eye(3)
        return result

    def jacobian_bias(self):
        return np.eye(6)

    def availability(self):
        return self.enabled
