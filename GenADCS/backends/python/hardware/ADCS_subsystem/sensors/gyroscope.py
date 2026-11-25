"""Compiled three-axis gyroscope measurement model."""

import numpy as np

from GenADCS.backends.python.helpers.random.philox import next_draw

from .kernels import rate_measurement
from .sensor import Sensor


class Gyroscope:
    """Three-axis rate sensor with additive body-frame bias and covariance."""

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

    def noiseless_biasfree_measurement(self, angular_velocity_body):
        return rate_measurement(
            angular_velocity_body,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            False,
            False,
        )

    def noiseless_biased_measurement(self, angular_velocity_body):
        return rate_measurement(
            angular_velocity_body,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            True,
            False,
        )

    def noisy_biasfree_measurement(self, angular_velocity_body):
        return rate_measurement(
            angular_velocity_body,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            False,
            True,
            next_draw(self, 3),
        )

    def noisy_biased_measurement(self, angular_velocity_body):
        return rate_measurement(
            angular_velocity_body,
            self.orientation,
            self.misalignment,
            self.bias,
            self.noise_cholesky,
            self.saturation_value,
            True,
            True,
            next_draw(self, 3),
        )

    def jacobian_angular_velocity(self):
        return np.eye(3)

    def jacobian_bias(self):
        return np.eye(3)

    def availability(self):
        return self.enabled
