__all__ = ["MTM"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, rot_mat

class MTM(Sensor):
    r"""
    Single–axis magnetometer (MTM) sensor model.

    This class implements a **single–axis magnetic field sensor** that measures
    the projection of the local geomagnetic field onto a specified body–fixed
    axis. It conforms to the generic sensor interface defined by
    :class:`~ADCS.satellite_hardware.sensor.Sensor`.

    Measurement model
    ------------------
    Let the geomagnetic field expressed in the spacecraft body frame be

    .. math::

        \mathbf{b}
        =
        \begin{bmatrix}
            b_x & b_y & b_z
        \end{bmatrix}^\top \in \mathbb{R}^3,

    and let the magnetometer sensitive axis be the unit vector
    :math:`\hat{\mathbf{a}} \in \mathbb{R}^3`.

    The clean (ideal) magnetometer measurement is

    .. math::

        y_{\text{clean}}
        =
        \mathbf{b}^\top \hat{\mathbf{a}}.

    Including bias and noise, the full measurement is

    .. math::

        z
        =
        \mathbf{b}^\top \hat{\mathbf{a}}
        + b + n,

    where:

    ======================= ==============================================
    Symbol                  Description
    ======================= ==============================================
    :math:`b`               Scalar magnetometer bias
    :math:`n`               Scalar measurement noise
    ======================= ==============================================

    The geomagnetic field vector :math:`\mathbf{b}` and its Jacobian with respect
    to the system state are provided by
    :class:`~ADCS.orbits.orbital_state.Orbital_State`.

    :param axis: Body–frame sensitive axis of the magnetometer, shape ``(3,)``.
                     The axis is normalized internally to unit length.
    :type axis: numpy.ndarray
    :param sample_time: Sampling period of the magnetometer in seconds.
    :type sample_time: float
    :param bias: Magnetometer bias model. If ``None``, a zero-bias model is used.
    :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
    :param noise: Magnetometer noise model. If ``None``, a zero-noise model is used.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
    :param estimate_bias: Indicates whether the magnetometer bias is included as part of the estimated filter state.
    :type estimate_bias: bool
    :return: None
    :rtype: None

    Notes
    -----
    * This is a **single–axis** sensor; therefore ``output_length = 1``.
    * The magnetometer itself is not an attitude sensor, but it provides
      attitude information when combined with an orbital magnetic field model.
    """
    def __init__(self, axis: np.ndarray,  sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        r"""
        Initialize the magnetometer sensor model.

        :param axis: Body–frame sensitive axis of the magnetometer, shape ``(3,)``.
                     The axis is normalized internally to unit length.
        :type axis: numpy.ndarray
        :param sample_time: Sampling period of the magnetometer in seconds.
        :type sample_time: float
        :param bias: Magnetometer bias model. If ``None``, a zero-bias model is used.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Magnetometer noise model. If ``None``, a zero-noise model is used.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: Indicates whether the magnetometer bias is included as part of the estimated filter state.
        :type estimate_bias: bool
        :return: None
        :rtype: None
        """
        self.axis = normalize(axis)
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean (bias– and noise–free) magnetometer measurement.

        The geomagnetic field vector in the body frame is obtained from the
        orbital state via

        .. math::

            \mathbf{b} = \mathbf{b}(\mathbf{x}, t),

        and projected onto the magnetometer axis:

        .. math::

            y_{\text{clean}}
            =
            \mathbf{b}^\top \hat{\mathbf{a}}.

        The magnetic field vector is expected to be provided by

        .. math::

            \texttt{os.get_state_vector(x)["b"]},

        where :math:`\mathbf{b} \in \mathbb{R}^3`.

        :param x: Full system state vector. This includes attitude and any additional
                  states required by the magnetic field model.
        :type x: numpy.ndarray
        :param os: Orbital and environmental model providing the geomagnetic field
                   in the body frame.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Clean magnetometer measurement along the sensor axis.
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        return np.dot(vecs["b"], self.axis)
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the magnetometer measurement with respect to the bias state.

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
        :type x: numpy.ndarray
        :param os: Orbital state object (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: ``(1, 1)`` Jacobian matrix if a bias model exists;
                 otherwise a ``(0, 1)`` empty matrix.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the magnetometer measurement with respect to the base
        (non-bias) system states.

        The clean measurement is

        .. math::

            y_{\text{clean}} = \mathbf{b}^\top \hat{\mathbf{a}}.

        Taking the derivative with respect to the base state
        :math:`\mathbf{x}_{\text{base}}` yields

        .. math::

            \frac{\partial y_{\text{clean}}}{\partial \mathbf{x}_{\text{base}}}
            =
            \left(
            \frac{\partial \mathbf{b}}{\partial \mathbf{x}_{\text{base}}}
            \right)^\top
            \hat{\mathbf{a}}.

        The orbital state is expected to provide the magnetic field Jacobian via

        .. math::

            \texttt{os.get_state_vector(x)["db"]}
            =
            \frac{\partial \mathbf{b}}{\partial \mathbf{x}_{\text{base}}}
            \in \mathbb{R}^{3 \times n_{\text{base}}}.

        The full base–state Jacobian returned by this method is

        .. math::

            \mathbf{H}_x
            =
            \begin{bmatrix}
                \mathbf{0}_{3 \times 1} \\
                \left(
                \frac{\partial \mathbf{b}}{\partial \mathbf{x}_{\text{base}}}
                \hat{\mathbf{a}}
                \right)
            \end{bmatrix}.

        :param x: Full system state vector.
        :type x: numpy.ndarray
        :param os: Orbital and environmental model providing both the geomagnetic
                   field and its Jacobian with respect to the base states.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Base–state Jacobian of the magnetometer measurement with respect
                 to the non-bias system states.
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        return np.vstack([np.zeros((3,1)), vecs['db']@np.expand_dims(self.axis,1)])
        
    

