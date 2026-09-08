"""Unscented attitude estimator with augmented parameter blocks."""

from .attitude_UKF import UKF

__all__ = ["AugmentedUKF"]


class AugmentedUKF(UKF):
    """UKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
