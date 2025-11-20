__all__ = ["MTM"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, rot_mat

class MTM(Sensor):
    r"""
    Magnetometer (MTM) sensor model.

    This sensor measures the component of the local magnetic field along a
    specified body–fixed axis. The *clean* (noise– and bias–free) measurement
    is given by

    .. math::

        y = \mathbf{b}^\top \hat{\mathbf{a}},

    where

    * :math:`\mathbf{b} \in \mathbb{R}^3` is the magnetic field vector in the
      body frame, and
    * :math:`\hat{\mathbf{a}} \in \mathbb{R}^3` is a **unit vector**
      representing the sensitive axis of the magnetometer.

    When combined with :class:`ADCS.orbits.orbital_state.Orbital_State`, this
    class can also provide Jacobians for use in extended Kalman filters (EKF).

    Parameters
    ----------
    axis : numpy.ndarray
        Body–frame axis of the magnetometer, shape ``(3,)``.
        This will be internally normalized to unit length.
    sample_time : float, optional
        Sampling period of the sensor in seconds (default: ``0.1`` s).
    bias : Bias, optional
        Bias model associated with the sensor. If provided, the bias state
        will be added to the base state and propagated according to
        :class:`ADCS.satellite_hardware.actuators.Bias`.
    noise : Noise, optional
        Noise model associated with the sensor. If provided, the measurement
        will be corrupted according to :class:`ADCS.satellite_hardware.actuators.Noise`.
    estimate_bias : bool, optional
        If ``True``, the bias is included as an estimated state for filtering
        (e.g. EKF). If ``False``, the bias is treated as known (or zero).

    Notes
    -----
    * ``output_length`` is set to ``1`` because this is a single–axis
      magnetometer.
    * The attribute :attr:`axis` is always stored as a normalized vector,
      regardless of the magnitude of the input ``axis``.
    """
    def __init__(self, axis: np.ndarray,  sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, scale: float = 1, estimate_bias: bool = False):
        self.axis = normalize(axis)
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, scale=scale, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the bias– and noise–free magnetometer measurement.

        This method evaluates

        .. math::

            y = \mathbf{b}^\top \hat{\mathbf{a}},

        where :math:`\mathbf{b}` is obtained from the orbital state and
        expressed in the body frame.

        Parameters
        ----------
        x : numpy.ndarray
            Current full state vector of the system (satellite attitude,
            angular velocity, and any additional states required by
            :class:`ADCS.orbits.orbital_state.Orbital_State`).
        os : Orbital_State
            Orbital and environmental model used to compute the magnetic field.
            The call ``os.get_state_vector(x=x)`` must return a dictionary with
            key ``"b"`` corresponding to the body–frame magnetic field
            :math:`\mathbf{b}`.

        Returns
        -------
        numpy.ndarray
            Clean measurement of the magnetic field along :attr:`axis`,
            shape ``(1,)``.
        """
        vecs = os.get_state_vector(x=x)
        return np.dot(vecs["b"], self.axis)
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        """
        Jacobian of the measurement with respect to the sensor bias state.

        If a bias model is present, the measurement is assumed to be

        .. math::

            z = y + b,

        where :math:`b` is a scalar bias. Hence

        .. math::

            \frac{\partial z}{\partial b} = 1.

        If no bias is present, an empty Jacobian is returned.

        Parameters
        ----------
        x : numpy.ndarray
            Current full state vector (unused here, included for interface
            consistency).
        os : Orbital_State
            Orbital state object (unused here, included for interface
            consistency).

        Returns
        -------
        numpy.ndarray
            Bias Jacobian matrix:

            * shape ``(1, 1)`` with value ``1`` if ``self.bias`` is not
              ``None``;
            * shape ``(0, 1)`` (empty) otherwise.
        """
        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the base (non–bias) states.

        The base–state Jacobian is defined as

        .. math::

            \frac{\partial y}{\partial \mathbf{x}_\text{base}}
            =
            \begin{bmatrix}
                \mathbf{0}_{3 \times 1} \\
                \frac{\partial \mathbf{b}}{\partial \mathbf{x}_\text{base}}
                \hat{\mathbf{a}}
            \end{bmatrix},

        where :math:`\mathbf{b}` is the body–frame magnetic field and
        :math:`\hat{\mathbf{a}}` is the unit axis of the MTM.

        This assumes that

        .. code-block:: python

            vecs = os.get_state_vector(x=x)
            vecs["db"]

        returns :math:`\partial \mathbf{b} / \partial \mathbf{x}_\text{base}`
        with shape ``(3, n_base)``. Post–multiplication by ``axis`` yields a
        column vector of length ``n_base``.

        Parameters
        ----------
        x : numpy.ndarray
            Current full state vector of the system.
        os : Orbital_State
            Orbital and environmental model providing both the magnetic field
            and its Jacobian with respect to the base states. Must supply
            keys ``"b"`` and ``"db"`` in the dictionary returned by
            :meth:`Orbital_State.get_state_vector`.

        Returns
        -------
        numpy.ndarray
            Jacobian of the measurement with respect to the base states.
            The returned array has shape ``(n_state_base, 1)``, where the first
            three rows are zeros and the remaining rows correspond to
            :math:`\partial \mathbf{b} / \partial \mathbf{x}_\text{base}
            \; \hat{\mathbf{a}}`.
        """
        vecs = os.get_state_vector(x=x)
        return np.vstack([np.zeros((3,1)), vecs['db']@np.expand_dims(self.axis,1)])
        
    

