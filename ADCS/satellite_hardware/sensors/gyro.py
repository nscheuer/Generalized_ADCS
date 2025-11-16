__all__ = ["Gyro"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize

class Gyro(Sensor):
    r"""
    Single–axis gyroscope sensor model.

    This sensor measures the component of the satellite's angular velocity
    vector :math:`\boldsymbol{\omega}` along a specified body–fixed axis.
    The *clean* (noise– and bias–free) measurement is

    .. math::

        y = \boldsymbol{\omega}^\top \hat{\mathbf{a}},

    where :math:`\hat{\mathbf{a}}` is the unit measurement axis.

    Parameters
    ----------
    axis : numpy.ndarray
        Body–frame sensing axis of the gyroscope, shape ``(3,)``.
        Automatically normalized to unit length.
    sample_time : float, optional
        Sensor sampling period in seconds (default: ``0.1``).
    bias : Bias, optional
        Bias model for the sensor (if present). Treated as an additive scalar
        bias to the measurement.
    noise : Noise, optional
        Noise model specifying additive measurement noise statistics.
    estimate_bias : bool, optional
        If ``True``, includes the bias as an estimated state for filtering
        (e.g., in an EKF). Otherwise the bias is treated as fixed or zero.
    """

    def __init__(self, axis: np.ndarray, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        self.axis = normalize(axis)
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the bias– and noise–free gyroscope measurement.

        This method extracts the angular velocity

        .. math:: \boldsymbol{\omega} = x_{0:3}

        from the satellite state vector ``x`` and computes its projection onto
        the sensor axis:

        .. math::

            y = \boldsymbol{\omega}^\top \hat{\mathbf{a}}.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector. The first three entries must be the
            body–frame angular velocity :math:`\boldsymbol{\omega}`.
        os : Orbital_State
            Unused for gyroscope measurements (provided for interface
            consistency).

        Returns
        -------
        numpy.ndarray
            Clean single–axis angular rate measurement, shape ``(1,)``.
        """
        return np.dot(x[0:3], self.axis)
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the gyroscope bias.

        If a bias model is present, the measurement is modeled as

        .. math:: z = y + b,

        with scalar bias :math:`b`. Therefore

        .. math:: \frac{\partial z}{\partial b} = 1.

        If no bias is included, returns an empty Jacobian.

        Parameters
        ----------
        x : numpy.ndarray
            Current full state vector (unused).
        os : Orbital_State
            Orbital state object (unused).

        Returns
        -------
        numpy.ndarray
            A ``(1,1)`` matrix containing ``1`` if bias is present,
            or a ``(0,1)`` empty matrix otherwise.
        """
        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean gyroscope measurement with respect to the
        base states (non–bias states).

        The measurement depends only on angular velocity:

        .. math::

            y = \boldsymbol{\omega}^\top \hat{\mathbf{a}},

        so

        .. math::

            \frac{\partial y}{\partial \boldsymbol{\omega}}
            = \hat{\mathbf{a}}, \qquad
            \frac{\partial y}{\partial q} = \mathbf{0}_{4 \times 1}.

        Thus, the Jacobian returned is

        .. math::

            \begin{bmatrix}
                \hat{\mathbf{a}} \\
                \mathbf{0}_{4 \times 1}
            \end{bmatrix}.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector.
        os : Orbital_State
            Orbital state object (unused).

        Returns
        -------
        numpy.ndarray
            Base–state Jacobian of shape ``(7, 1)``, consisting of the
            3-component gyroscope axis followed by four zeros.
        """
        return np.vstack([self.axis.reshape((3, 1)), np.zeros((4,1))])

