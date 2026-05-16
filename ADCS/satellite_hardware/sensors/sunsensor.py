__all__ = ["SunSensor"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, rot_mat

class SunSensor(Sensor):
    r"""
    Single-axis coarse Sun sensor model.

    This class implements a **single-axis coarse Sun sensor** that measures the
    cosine of the incidence angle between a fixed body-frame sensor axis and the
    Sun direction. The measurement is scaled by a constant efficiency factor and
    clipped to zero when the Sun is outside the sensor’s field of view.

    The sensor follows the generic interface defined by
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`.

    Measurement model
    -----------------
    Let

    ============================== ============================================
    Symbol                          Description
    ============================== ============================================
    :math:`\hat{\mathbf{a}}`        Unit sensor axis (body frame)
    :math:`\hat{\mathbf{s}}`        Unit Sun direction (body frame)
    :math:`\eta`                   Scalar sensor efficiency
    ============================== ============================================

    The Sun direction expressed in the body frame is obtained from the orbital
    state as

    .. math::

        \hat{\mathbf{s}}
        =
        \frac{\mathbf{s} - \mathbf{r}}
             {\lVert \mathbf{s} - \mathbf{r} \rVert},

    where :math:`\mathbf{r}` is the spacecraft position and :math:`\mathbf{s}` is
    the Sun position, both provided by
    :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.

    The cosine of the incidence angle between the Sun and the sensor axis is

    .. math::

        c = \hat{\mathbf{a}}^\top \hat{\mathbf{s}}.

    The clean (noise- and bias-free) measurement is defined as

    .. math::

        y_{\text{clean}}
        =
        \begin{cases}
            \eta \, \max(c, 0), & \text{if } \texttt{os.is\_sunlit()} = \text{True}, \\
            0,                 & \text{if } \texttt{os.is\_sunlit()} = \text{False}.
        \end{cases}

    Measurement with errors
    -----------------------
    When bias and noise models are present, the full measurement is

    .. math::

        z = y_{\text{clean}} + b + n,

    where

    * :math:`b` is an additive scalar bias modeled by
      :class:`~ADCS.satellite_hardware.errors.bias.Bias`,
    * :math:`n` is additive noise modeled by
      :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

    Notes
    -----
    * This is a **coarse** Sun sensor intended for low-accuracy attitude
      determination.
    * The sensor output is scalar, so ``output_length = 1``.
    * In eclipse (``os.is_sunlit() == False``) the clean measurement and all
      Jacobians return ``NaN`` as the "sensor invalid / occulted" sentinel;
      the UKF/SRUKF detect this (``np.isnan``) and deactivate the sensor for
      that step. (It is *not* zero -- a zero would be fed to the filter as a
      real measurement.)
    """
    def __init__(self, axis: np.ndarray, efficiency: float, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        r"""
        Initialize a single-axis coarse Sun sensor.

        :param axis: Body-frame sensor axis. This vector is normalized internally.
        :type axis: numpy.ndarray, shape ``(3,)``
        :param efficiency: Scalar efficiency gain applied to the illuminated
            portion of the measurement.
        :type efficiency: float
        :param sample_time: Sampling period of the sensor in seconds.
        :type sample_time: float
        :param bias: Additive bias model applied to the measurement.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Additive noise model applied to the measurement.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: Flag indicating whether the bias is included in the
            estimator state.
        :type estimate_bias: bool

        :return: None
        :rtype: None
        """
        self.axis = normalize(axis)
        self.efficiency = efficiency
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean (noise- and bias-free) Sun sensor measurement.

        The Sun direction is computed from the orbital state as

        .. math::

            \hat{\mathbf{s}}
            =
            \frac{\mathbf{s} - \mathbf{r}}
                 {\lVert \mathbf{s} - \mathbf{r} \rVert},

        and the cosine incidence term is

        .. math::

            c = \hat{\mathbf{a}}^\top \hat{\mathbf{s}}.

        The illuminated contribution is defined by

        .. math::

            \text{illumination} = \max(c, 0).

        The resulting clean measurement is

        .. math::

            y_{\text{clean}}
            =
            \begin{cases}
                \eta \, \text{illumination}, & \text{if sunlit}, \\
                0,                           & \text{if in eclipse}.
            \end{cases}

        :param x: Full system state vector.
        :type x: numpy.ndarray
        :param os: Orbital state providing spacecraft position, Sun position,
            and lighting conditions via
            :meth:`~ADCS.orbits.orbital_state.Orbital_State.is_sunlit`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Clean Sun sensor measurement of shape ``(1,)``. If the spacecraft
            is not sunlit, the value is defined to be ``NaN``.
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        sun_dir = normalize(vecs["s"] - vecs["r"])

        # Compute illumination
        projection = np.dot(self.axis, sun_dir)
        illumination = np.maximum(projection, 0.0)

        # Eclipse handling
        if os.is_sunlit():
            return self.efficiency * illumination
        else:
            return np.nan
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the Sun sensor bias.

        If a bias model is present, the measurement equation is

        .. math::

            z = y + b,

        where :math:`b` is a scalar bias. The corresponding Jacobian is

        .. math::

            \frac{\partial z}{\partial b} = 1.

        If no bias model exists, an empty Jacobian is returned.

        :param x: Full system state vector (unused).
        :type x: numpy.ndarray
        :param os: Orbital state object (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: A ``(1,1)`` array containing ``1`` if a bias exists, otherwise a
            ``(0,1)`` empty array.
        :rtype: numpy.ndarray
        """

        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean Sun sensor measurement with respect to the base
        ADCS state.

        The base state is defined as

        .. math::

            \mathbf{x}
            =
            [\omega_x, \omega_y, \omega_z, q_0, q_1, q_2, q_3]^\top,

        where :math:`\boldsymbol{\omega}` is the body angular rate and
        :math:`\mathbf{q}` is the attitude quaternion.

        The Sun direction is

        .. math::

            \hat{\mathbf{s}}
            =
            \frac{\mathbf{s} - \mathbf{r}}
                 {\lVert \mathbf{s} - \mathbf{r} \rVert},

        and the cosine incidence is

        .. math::

            c = \hat{\mathbf{a}}^\top \hat{\mathbf{s}}.

        The derivative of the normalized Sun vector with respect to the state is
        computed using :func:`~ADCS.helpers.math_helpers.normed_vec_jac`, yielding

        .. math::

            \frac{\partial \hat{\mathbf{s}}}{\partial \mathbf{x}}.

        The derivative of the cosine term is then

        .. math::

            \frac{\partial c}{\partial \mathbf{x}}
            =
            \left(
                \frac{\partial \hat{\mathbf{s}}}{\partial \mathbf{x}}
            \right)^\top
            \hat{\mathbf{a}}.

        Because the measurement uses :math:`\max(c, 0)`, the Jacobian is defined
        to be zero whenever :math:`c \le 0`. Furthermore, when the spacecraft is
        in eclipse (``os.is_sunlit() == False``), the measurement is identically
        zero and all partial derivatives vanish.

        :param x: Full 7-element ADCS state vector.
        :type x: numpy.ndarray
        :param os: Orbital state providing Sun/spacecraft geometry and lighting
            information.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Jacobian of the clean measurement with respect to the base
            state, with shape ``(7, 1)``. The Jacobian is zero if the spacecraft
            is in eclipse or if the Sun is behind the sensor.
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        sunvec = vecs["s"] - vecs["r"]
        ns = normalize(sunvec)

        dns__dq = normed_vec_jac(sunvec,vecs["ds"]-vecs["dr"])
        cos_incidence = np.dot(ns, self.axis)
        dcos_incidence__dq = (cos_incidence>0)*(dns__dq@self.axis)
        if os.is_sunlit():
            return np.vstack([np.zeros((3,1)),self.efficiency*np.expand_dims(dcos_incidence__dq,1)])
        else:
            # In eclipse clean_reading() returns NaN as the "sensor invalid /
            # occulted" sentinel that the UKF/SRUKF use (np.isnan check) to
            # deactivate this sensor. The Jacobian must carry the SAME sentinel
            # for a consistent contract: returning finite zeros silently
            # asserted "constant 0 measurement, zero sensitivity", which would
            # bias the filter toward believing the eclipse reading is exactly 0
            # if it were ever used without the active mask.
            return np.full((7, 1), np.nan)
        
    

