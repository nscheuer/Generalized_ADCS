__all__ = ["SunPair"]

from .sensor import Sensor

import numpy as np
from typing import Tuple
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import normalize, rot_mat


class SunPair(Sensor):
    r"""
    Dual-hemisphere coarse Sun sensor model.

    This class models a **pair of opposing coarse Sun sensors** aligned along a
    single body-frame axis. The two sensors together provide a **single scalar
    measurement** proportional to the projection of the Sun direction onto the
    sensor axis, scaled by a hemisphere-dependent efficiency.

    The class conforms to the generic sensor interface defined by
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`.

    Conceptual model
    ----------------
    Let

    ============================== ============================================
    Symbol                          Description
    ============================== ============================================
    :math:`\hat{\mathbf{a}}`        Unit sensor axis (body frame)
    :math:`\hat{\mathbf{s}}`        Unit Sun direction (body frame)
    :math:`\eta_\text{front}`       Efficiency for +axis hemisphere
    :math:`\eta_\text{back}`        Efficiency for −axis hemisphere
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

    The clean (ideal) measurement is defined as

    .. math::

        y_{\text{clean}}
        =
        \begin{cases}
            (\hat{\mathbf{a}}^\top \hat{\mathbf{s}})\,\eta_\text{front},
            & \hat{\mathbf{a}}^\top \hat{\mathbf{s}} > 0, \\
            (\hat{\mathbf{a}}^\top \hat{\mathbf{s}})\,\eta_\text{back},
            & \hat{\mathbf{a}}^\top \hat{\mathbf{s}} \le 0.
        \end{cases}

    If the spacecraft is not illuminated by the Sun, i.e.,

    .. math::

        \texttt{os.is\_sunlit()} = \text{False},

    the measurement is undefined and the sensor returns ``NaN``.

    Measurement with errors
    -----------------------
    Including bias and noise, the full sensor output is

    .. math::

        z = y_{\text{clean}} + b + n,

    where

    * :math:`b` is an additive scalar bias modeled by
      :class:`~ADCS.satellite_hardware.errors.bias.Bias`,
    * :math:`n` is additive noise modeled by
      :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

    Notes
    -----
    * This is a **coarse Sun sensor**, intended for low-accuracy attitude
      determination.
    * The output dimension is scalar, so ``output_length = 1``.
    * In eclipse the clean measurement and all Jacobians return ``NaN`` as the
      "sensor invalid / occulted" sentinel; the UKF/SRUKF detect this
      (``np.isnan``) and deactivate the sensor (it is *not* zero).
    """

    def __init__(
        self,
        axis: np.ndarray,
        efficiency: Tuple[float, float],
        sample_time: float = 0.1,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ):
        r"""
        Initialize a dual-hemisphere coarse Sun sensor.

        :param axis: Body-frame sensor axis. This vector is normalized internally.
        :type axis: numpy.ndarray, shape ``(3,)``
        :param efficiency: Tuple ``(front, back)`` specifying efficiency
            coefficients for the +axis and −axis hemispheres. If a single scalar
            is provided, it is used for both hemispheres.
        :type efficiency: tuple[float, float]
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
        if isinstance(efficiency, tuple):
            self.efficiency = efficiency # (front, back)
        else:
            self.efficiency = (efficiency, efficiency)
        self.attitude_sensor = False

        super().__init__(
            sample_time=sample_time,
            output_length=1,
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias,
        )

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean (noise- and bias-free) Sun sensor measurement.

        The clean measurement is computed according to the dual-hemisphere model
        described in the class documentation. If the spacecraft is in eclipse
        (``os.is_sunlit() == False``), the clean measurement is undefined and
        ``NaN`` is returned.

        :param x: Full system state vector.
        :type x: numpy.ndarray
        :param os: Orbital state providing spacecraft position, Sun position,
            and lighting conditions.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Clean scalar Sun-sensor measurement wrapped in a 1-element array,
            or ``NaN`` if the spacecraft is not sunlit.
        :rtype: numpy.ndarray
        """

        if not os.is_sunlit():
            return np.nan

        vecs = os.get_state_vector(x=x)

        sun_dir = normalize(vecs["s"] - vecs["r"])

        proj = float(np.dot(self.axis, sun_dir))
        eff = self.efficiency[0] if proj > 0.0 else self.efficiency[1]
        return proj * eff

    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the Sun-sensor bias.

        When a bias model is present, the measurement model is

        .. math::

            z = y + b,

        where :math:`b` is a scalar bias. The Jacobian is therefore

        .. math::

            \frac{\partial z}{\partial b} = 1.

        If no bias model exists, an empty Jacobian is returned.

        :param x: Full system state vector (unused).
        :type x: numpy.ndarray
        :param os: Orbital state (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: A ``(1,1)`` identity matrix if a bias exists, otherwise a
            ``(0,1)`` empty array.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return np.ones((1, 1))
        else:
            return np.zeros((0, 1))

    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean Sun-sensor measurement with respect to the base
        ADCS state.

        The base state is defined as

        .. math::

            \mathbf{x}
            =
            [\omega_x, \omega_y, \omega_z, q_0, q_1, q_2, q_3]^\top,

        where :math:`\boldsymbol{\omega}` is the body angular rate and
        :math:`\mathbf{q}` is the attitude quaternion.

        Because the Sun direction in the body frame depends on spacecraft
        attitude, this Jacobian is computed numerically using **central finite
        differences**:

        .. math::

            \frac{\partial y}{\partial x_i}
            \approx
            \frac{f(x_i + \varepsilon) - f(x_i - \varepsilon)}{2\varepsilon},

        with :math:`\varepsilon = 10^{-6}` and :math:`f(\cdot)` defined by
        :meth:`_clean_scalar`.

        If the spacecraft is not sunlit, the clean measurement is treated as
        invalid and this method returns a ``NaN`` sentinel rather than a finite
        Jacobian.

        :param x: Full 7-element ADCS state vector.
        :type x: numpy.ndarray
        :param os: Orbital state providing lighting conditions and Sun/spacecraft
            geometry.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Jacobian of the clean measurement with respect to the base
            state, with shape ``(7, 1)``. Returns ``NaN`` in eclipse.
        :rtype: numpy.ndarray
        """
        # In eclipse clean_reading() returns NaN as the "sensor invalid /
        # occulted" sentinel the UKF/SRUKF use to deactivate this sensor; the
        # Jacobian must carry the SAME sentinel (finite zeros silently asserted
        # a constant-0 measurement with zero sensitivity).
        if not os.is_sunlit():
            return np.full((7, 1), np.nan)

        eps = 1e-6
        grad = np.zeros(7)

        f0 = self._clean_scalar(x, os)

        for i in range(7):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            fp = self._clean_scalar(xp, os)
            fm = self._clean_scalar(xm, os)
            grad[i] = (fp - fm) / (2.0 * eps)

        return grad.reshape(7, 1)
    
    def _clean_scalar(self, x: np.ndarray, os: Orbital_State) -> float:
        r"""
        Compute the clean Sun-sensor measurement as a scalar.

        This helper method implements the same clean measurement model as
        :meth:`clean_reading`, but returns a scalar value. It is primarily used
        internally for numerical differentiation when computing Jacobians.

        If the spacecraft is not sunlit, the returned value is defined to be
        zero.

        :param x: Full system state vector.
        :type x: numpy.ndarray
        :param os: Orbital state providing spacecraft position, Sun position,
            and lighting conditions.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Clean scalar Sun-sensor measurement.
        :rtype: float
        """
        if not os.is_sunlit():
            return 0.0

        vecs = os.get_state_vector(x=x)

        sun_dir = normalize(vecs["s"] - vecs["r"])

        proj = float(np.dot(self.axis, sun_dir))
        eff = self.efficiency[0] if proj > 0.0 else self.efficiency[1]
        return proj * eff
