__all__ = ["Gyro"]

from .sensor import Sensor

import numpy as np

from ADCS.state import State
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from typing import Optional

class Gyro(Sensor):
    r"""
    Single–axis gyroscope sensor model.

    This class implements a body–fixed **single–axis angular rate sensor** that
    measures the projection of the spacecraft angular velocity vector onto a
    specified measurement axis.

    The gyroscope model follows the generic sensor interface defined in
    :class:`~ADCS.satellite_hardware.sensor.Sensor`.

    Measurement model
    ------------------
    Let the spacecraft angular velocity expressed in the body frame be

    .. math::

        \boldsymbol{\omega}
        =
        \begin{bmatrix}
            \omega_x & \omega_y & \omega_z
        \end{bmatrix}^\top.

    The clean (ideal) gyroscope measurement is

    .. math::

        y_{\text{clean}}
        =
        \boldsymbol{\omega}^\top \hat{\mathbf{a}},

    where :math:`\hat{\mathbf{a}}` is the unit measurement axis of the gyroscope.

    Including bias and noise, the full measurement is

    .. math::

        z
        =
        \boldsymbol{\omega}^\top \hat{\mathbf{a}}
        + b + n,

    with:

    ======================= ==============================================
    Symbol                  Description
    ======================= ==============================================
    :math:`b`               Scalar gyroscope bias
    :math:`n`               Scalar measurement noise
    ======================= ==============================================

    Bias and noise are modeled using
    :class:`~ADCS.satellite_hardware.errors.bias.Bias` and
    :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

    :param axis: Body–frame measurement axis, shape ``(3,)``.
                     The axis is normalized internally to ensure unit magnitude.
    :type axis: numpy.ndarray
    :param sample_time: Sampling period of the gyroscope in seconds.
    :type sample_time: float
    :param bias: Gyroscope bias model. If ``None``, a zero-bias model is used.
    :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
    :param noise: Gyroscope noise model. If ``None``, a zero-noise model is used.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
    :param estimate_bias: Indicates whether the gyroscope bias is included in the estimated filter state.
    :type estimate_bias: bool
    :return: None
    :rtype: None

    Notes
    -----
    * This sensor measures **angular rate only** and is not an attitude sensor.
    * The angular velocity is assumed to occupy the first three elements of the
      system state vector.
    """


    def __init__(self, axis: np.ndarray, sample_time: float = 0.1, bias: Optional[Bias] = None, noise: Optional[Noise] = None, estimate_bias: bool = False):
        r"""
        Initialize the single–axis gyroscope sensor.

        :param axis: Body–frame measurement axis, shape ``(3,)``.
                     The axis is normalized internally to ensure unit magnitude.
        :type axis: numpy.ndarray
        :param sample_time: Sampling period of the gyroscope in seconds.
        :type sample_time: float
        :param bias: Gyroscope bias model. If ``None``, a zero-bias model is used.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Gyroscope noise model. If ``None``, a zero-noise model is used.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: Indicates whether the gyroscope bias is included in the estimated filter state.
        :type estimate_bias: bool
        :return: None
        :rtype: None
        """
        self.axis = normalize(axis)
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean (bias– and noise–free) gyroscope measurement.

        The angular velocity is extracted from the system state vector as

        .. math::

            \boldsymbol{\omega} = x_{0:3},

        and projected onto the sensor axis:

        .. math::

            y_{\text{clean}}
            =
            \boldsymbol{\omega}^\top \hat{\mathbf{a}}.

        :param x: Full system state vector. The first three elements must correspond
                  to the body–frame angular velocity vector
                  :math:`\boldsymbol{\omega}`.
        :type x: ADCS.state.State
        :param os: Orbital state object. This argument is unused by the gyroscope and
                   is provided for interface consistency.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Clean single–axis angular rate measurement.
        :rtype: numpy.ndarray
        """
        return np.dot(x.w, self.axis)
    
    def bias_jac(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the gyroscope measurement with respect to the bias state.

        When a bias model is present, the measurement equation is

        .. math::

            z = y_{\text{clean}} + b,

        where :math:`b` is a scalar bias term.

        The Jacobian with respect to the bias is therefore

        .. math::

            \mathbf{H}_b
            =
            \frac{\partial z}{\partial b}
            =
            \begin{bmatrix}
                1
            \end{bmatrix}.

        If no bias model is included, an empty Jacobian is returned.

        :param x: Full system state vector (unused).
        :type x: ADCS.state.State
        :param os: Orbital state object (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: ``(1, 1)`` matrix containing ``1`` if a bias model exists;
                 otherwise a ``(0, 1)`` empty matrix.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean gyroscope measurement with respect to the
        base (non-bias) system states.

        The clean measurement depends only on angular velocity:

        .. math::

            y_{\text{clean}} = \boldsymbol{\omega}^\top \hat{\mathbf{a}}.

        Therefore,

        .. math::

            \frac{\partial y_{\text{clean}}}{\partial \boldsymbol{\omega}}
            =
            \hat{\mathbf{a}}, \qquad
            \frac{\partial y_{\text{clean}}}{\partial \mathbf{q}}
            =
            \mathbf{0}_{4 \times 1},

        where :math:`\mathbf{q}` is the attitude quaternion.

        The resulting Jacobian is

        .. math::

            \mathbf{H}_x
            =
            \begin{bmatrix}
                \hat{\mathbf{a}} \\
                \mathbf{0}_{4 \times 1}
            \end{bmatrix}
            \in \mathbb{R}^{7 \times 1}.

        :param x: Full system state vector.
        :type x: ADCS.state.State
        :param os: Orbital state object (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Base–state Jacobian matrix of shape ``(7, 1)``.
        :rtype: numpy.ndarray
        """
        return np.vstack([self.axis.reshape((3, 1)), np.zeros((4,1))])
