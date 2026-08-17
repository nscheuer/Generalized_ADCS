__all__ = ["Orbit_Estimator"]

import numpy as np
from typing import Optional, List, Sequence
from abc import ABC, abstractmethod

from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite


class Orbit_Estimator(ABC):
    r"""
    Abstract base class for orbit determination (OD) algorithms.

    This class defines the common *interface* and *conceptual model* for all orbit
    estimators used within the ADCS software stack. An orbit estimator is responsible
    for estimating the satellite orbital state vector in an inertial reference frame
    using a combination of dynamical propagation and sensor measurements.

    The estimated orbital state is defined as the 6-dimensional state vector

    .. math::

        \mathbf{x}
        =
        \begin{bmatrix}
        \mathbf{r} \\
        \mathbf{v}
        \end{bmatrix}
        \in \mathbb{R}^6

    where:

    * :math:`\mathbf{r} \in \mathbb{R}^3` is the position vector in the
      Earth-Centered Inertial (ECI) frame
    * :math:`\mathbf{v} \in \mathbb{R}^3` is the velocity vector in the same frame

    The estimator maintains a probabilistic belief over this state in the form of:

    * An estimated mean state :math:`\hat{\mathbf{x}}`
    * An associated covariance matrix :math:`\mathbf{P}`

    which are encapsulated by the
    :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State` data structure.

    This base class does **not** implement a specific estimation algorithm.
    Concrete subclasses (e.g. EKF, UKF, batch least squares) must implement the
    :meth:`update` and :meth:`reset` methods.

    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :type dt: float
    :type os_hat: :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State` or None

    """
    def __init__(self, est_sat: EstimatedSatellite, dt: float = 1.0) -> None:
        r"""
        Initialize the orbit estimator base attributes.

        This constructor sets up the estimator with the satellite hardware model
        and the discrete propagation time step. No state estimate is created at
        initialization time; the estimator must be initialized explicitly via
        :meth:`reset` or during the first call to :meth:`update`.

        The satellite hardware model is used to retrieve sensor configuration and
        measurement noise characteristics, which typically define the measurement
        covariance matrix :math:`\mathbf{R}`.

        :param est_sat:
            Estimated satellite hardware model containing sensor definitions and
            noise parameters.
        :type est_sat:
            :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

        :param dt:
            Discrete propagation time step :math:`\Delta t` in seconds used by the
            estimator during time updates.
        :type dt:
            float

        :return:
            None
        :rtype:
            None

        """
        self.est_sat = est_sat
        self.dt = dt

        self.os_hat: Optional[EstimatedOrbital_State] = None

    @abstractmethod
    def update(self, GPS_measurements: Sequence[np.ndarray], J2000: float) -> EstimatedOrbital_State:
        r"""
        Propagate and update the orbital state estimate using sensor measurements.

        This method performs one full estimation cycle consisting of:

        1. **Time Update (Prediction)**  
           The previous state estimate :math:`\hat{\mathbf{x}}_{k-1}` is propagated
           forward in time according to the orbital dynamics model:

           .. math::

               \hat{\mathbf{x}}_k^{-}
               =
               f\!\left(\hat{\mathbf{x}}_{k-1}, \Delta t\right)

        2. **Measurement Update (Correction)**  
           The predicted state is corrected using available GPS measurements
           :math:`\mathbf{z}_k`:

           .. math::

               \hat{\mathbf{x}}_k
               =
               \hat{\mathbf{x}}_k^{-}
               +
               \mathbf{K}_k
               \left(
               \mathbf{z}_k - h\!\left(\hat{\mathbf{x}}_k^{-}\right)
               \right)

        where:

        * :math:`f(\cdot)` is the nonlinear orbital dynamics model
        * :math:`h(\cdot)` is the measurement model
        * :math:`\mathbf{K}_k` is the estimator-specific gain matrix

        The measurement list may contain multiple GPS observations at the same epoch,
        each expressed either as:

        * Position-only measurements: :math:`\mathbf{z}_k \in \mathbb{R}^3`
        * Position-velocity measurements: :math:`\mathbf{z}_k \in \mathbb{R}^6`

        All measurements are assumed to be expressed in the ECI frame and time-tagged
        at the provided J2000 epoch.

        :param GPS_measurements:
            List of GPS measurement vectors used to update the orbital state estimate.
        :type GPS_measurements:
            list[numpy.ndarray]

        :param J2000:
            Current epoch time expressed in seconds since J2000.
        :type J2000:
            float

        :return:
            Updated estimated orbital state containing the state vector
            :math:`\hat{\mathbf{x}}_k` and covariance matrix :math:`\mathbf{P}_k`.
        :rtype:
            :class:`~ADCS.estimators.estimator_helpers.EstimatedOrbital_State`

        """
        pass

    @abstractmethod
    def reset(self, **kwargs) -> None:
        r"""
        Reset the estimator internal state and covariance.

        This method re-initializes the estimator without requiring a new instance
        to be created. Concrete implementations should use this method to:

        * Define the initial state estimate :math:`\hat{\mathbf{x}}_0`
        * Initialize the covariance matrix :math:`\mathbf{P}_0`
        * Configure process noise models and tuning parameters

        Typical reset behavior may include:

        .. math::

            \hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}}_0,
            \qquad
            \mathbf{P} \leftarrow \mathbf{P}_0

        The exact parameters required depend on the specific estimator
        implementation (e.g. EKF vs. batch estimator) and are passed via keyword
        arguments.

        :param kwargs:
            Implementation-specific initialization parameters such as initial state
            vectors, covariance matrices, or noise tuning values.
        :type kwargs:
            dict

        :return:
            None
        :rtype:
            None

        """
        pass

    