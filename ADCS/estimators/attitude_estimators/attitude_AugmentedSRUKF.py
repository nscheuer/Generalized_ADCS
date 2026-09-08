"""Square-root unscented attitude estimator with augmented parameter blocks."""

from .attitude_SRUKF import SRUKF

__all__ = ["AugmentedSRUKF"]


class AugmentedSRUKF(SRUKF):
    """SRUKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
