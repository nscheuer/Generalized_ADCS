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

    This sensor consists of two photodiodes mounted on opposite sides of the
    same body-frame axis. A single scalar measurement is reported, equal to
    the projection of the Sun vector onto the axis, multiplied by an
    **efficiency factor** that depends on whether the Sun is in the forward
    or backward hemisphere.

    The clean measurement is

    .. math::

        y =
        \begin{cases}
            (\hat{\mathbf{a}}^\top \hat{\mathbf{s}})\,\eta_\text{front},
            & \hat{\mathbf{a}}^\top \hat{\mathbf{s}} > 0, \\
            (\hat{\mathbf{a}}^\top \hat{\mathbf{s}})\,\eta_\text{back},
            & \hat{\mathbf{a}}^\top \hat{\mathbf{s}} \le 0,
        \end{cases}

    where

    * :math:`\hat{\mathbf{a}}` — unit sensor axis,
    * :math:`\hat{\mathbf{s}}` — Sun direction in the body frame,
    * :math:`\eta_\text{front}, \eta_\text{back}` — the two efficiency values.

    When the spacecraft is in eclipse (``os.is_sunlit() == False``), the
    clean reading is zero and all Jacobians vanish.

    Parameters
    ----------
    axis : numpy.ndarray
        Body-frame sensor axis. It is internally normalized.
    efficiency : tuple of float
        ``(front, back)`` efficiency coefficients for the +axis and −axis
        hemispheres respectively.
    sample_time : float, optional
        Sampling period in seconds (default: ``0.1``).
    bias : Bias, optional
        Additive bias model for the measurement.
    noise : Noise, optional
        Additive noise model for the measurement.
    estimate_bias : bool, optional
        Whether the bias is included as an estimated state in filtering.
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
        Compute the clean (noise- and bias-free) Sun sensor reading.

        This returns a 1-element array containing the clean scalar value
        computed by :meth:`_clean_scalar`.

        If the spacecraft is in darkness (``os.is_sunlit() == False``),
        the clean reading is defined to be np.nan.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector.
        os : Orbital_State
            Orbital state that provides the spacecraft position, Sun position,
            and lighting conditions.

        Returns
        -------
        float
            Element containing the clean measurement.
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

        If a bias model is present, the measurement is

        .. math:: z = y + b,

        where :math:`b` is a scalar bias. Thus,

        .. math:: \frac{\partial z}{\partial b} = 1.

        If no bias model exists, a ``0×1`` empty Jacobian is returned.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : Orbital_State
            Orbital state (unused).

        Returns
        -------
        numpy.ndarray
            ``(1,1)`` identity if bias exists, otherwise ``(0,1)``.
        """
        if self.bias:
            return np.ones((1, 1))
        else:
            return np.zeros((0, 1))

    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        """
        Jacobian of the clean Sun-sensor measurement with respect to the base
        state

        .. math:: x = [\omega_x, \omega_y, \omega_z, q_0, q_1, q_2, q_3].

        Because the Sun direction in the body frame depends on spacecraft
        attitude, this derivative is computed numerically using **central
        finite differences**:

        .. math::

            \frac{\partial y}{\partial x_i}
            \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon},

        with :math:`\epsilon = 10^{-6}`.

        If the spacecraft is in darkness, the clean reading is identically
        zero and all partial derivatives are zero.

        Parameters
        ----------
        x : numpy.ndarray
            Full 7-element ADCS state vector.
        os : Orbital_State
            Orbital state providing lighting conditions and Sun/spacecraft
            geometry.

        Returns
        -------
        numpy.ndarray
            A ``(7,1)`` Jacobian vector :math:`\partial y / \partial x`.
        """
        # If dark, derivative is identically zero
        if not os.is_sunlit():
            return np.zeros((7, 1))

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
        """
        Clean reading as a scalar (helper to simplify finite-diff Jacobian).
        """
        if not os.is_sunlit():
            return 0.0

        vecs = os.get_state_vector(x=x)

        sun_dir = normalize(vecs["s"] - vecs["r"])

        proj = float(np.dot(self.axis, sun_dir))
        eff = self.efficiency[0] if proj > 0.0 else self.efficiency[1]
        return proj * eff