__all__ = ["Orbit_Estimator"]

import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod

from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Orbit_Estimator(ABC):
    r"""
    Abstract Base Class for Orbit Determination algorithms.

    This class defines the standard interface for estimators (filters) that determine 
    the satellite's orbital state vector :math:`\mathbf{x}`, defined as:

    .. math::
        \mathbf{x} = \begin{bmatrix} \mathbf{r} \\ \mathbf{v} \end{bmatrix} \in \mathbb{R}^6

    where :math:`\mathbf{r}` is the position vector and :math:`\mathbf{v}` is the velocity vector 
    in the Earth-Centered Inertial (ECI) frame.

    Attributes:
        est_sat (EstimatedSatellite): The satellite hardware model containing sensor noise parameters.
        dt (float): The propagation time step :math:`\Delta t` in seconds.
        os_hat (Optional[EstimatedOrbital_State]): The current estimated state :math:`\hat{\mathbf{x}}` 
            and covariance :math:`\mathbf{P}`. Initially ``None``.
    """
    def __init__(self, est_sat: EstimatedSatellite, dt: float = 1.0) -> None:
        r"""
        Initialize the Orbit Estimator base attributes.

        :param est_sat: The satellite hardware instance used to retrieve sensor configurations 
                        and noise characteristics :math:`\mathbf{R}`.
        :param dt: The time step :math:`\Delta t` [s] used for discrete time propagation. 
                   Defaults to 1.0.
        """
        self.est_sat = est_sat
        self.dt = dt

        self.os_hat: Optional[EstimatedOrbital_State] = None

    @abstractmethod
    def update(self, GPS_measurements: List[np.ndarray], J2000: float) -> EstimatedOrbital_State:
        r"""
        Propagate dynamics and update the state estimate using new sensor measurements.

        This method performs the core estimation cycle:
        1. **Time Update**: Propagate state :math:`\hat{\mathbf{x}}_{k-1} \to \hat{\mathbf{x}}_k^-`.
        2. **Measurement Update**: Correct the predicted state using measurement :math:`\mathbf{z}_k`.

        :param GPS_measurements: A list of measurement vectors :math:`\mathbf{z}_k` from available sensors.
                                 Can be Position-only (:math:`\mathbb{R}^3`) or PV (:math:`\mathbb{R}^6`).
        :param J2000: The current epoch time :math:`t_k` in J2000 seconds.
        
        :return: The updated estimated orbital state object containing :math:`\hat{\mathbf{x}}_k` 
                 and :math:`\mathbf{P}_k`.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs) -> None:
        r"""
        Reset the estimator internal state and covariance.

        This allows for re-initialization of the filter without instantiating a new object. 
        Specific implementations should handle the setup of initial covariance matrices 
        :math:`\mathbf{P}_0` and process noise :math:`\mathbf{Q}` here.

        :param kwargs: Keyword arguments specific to the implementation (e.g., initial state vector, 
                       specific covariance tuning parameters).
        """
        pass

    