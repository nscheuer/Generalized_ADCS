__all__ = ["SunSensor"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, rot_mat

class SunSensor(Sensor):
    r"""
    Single-axis coarse Sun sensor model.

    This sensor measures the cosine of the incidence angle between a fixed
    body-frame axis and the Sun direction, scaled by an efficiency factor.
    The *clean* (noise- and bias-free) measurement is

    .. math::

        y =
        \begin{cases}
            \eta \,\max(\hat{\mathbf{a}}^\top \hat{\mathbf{s}}, 0), & \text{if sunlit}, \\
            0, & \text{if in eclipse},
        \end{cases}

    where

    * :math:`\hat{\mathbf{a}}` — unit sensor axis in the body frame,
    * :math:`\hat{\mathbf{s}}` — unit Sun direction in the body frame,
    * :math:`\eta` — scalar sensor efficiency.

    In eclipse (``os.is_sunlit() == False``), the clean reading is forced to
    zero regardless of geometry.

    Parameters
    ----------
    axis : numpy.ndarray
        Body-frame sensor axis, shape ``(3,)``. It is internally normalized.
    efficiency : float
        Scalar efficiency gain applied to the illuminated portion of the signal.
    sample_time : float, optional
        Sampling period in seconds (default: ``0.1``).
    bias : Bias, optional
        Additive scalar bias model for the measurement. If not provided, a
        zero-bias model is used.
    noise : Noise, optional
        Additive scalar noise model for the measurement. If not provided, a
        zero-noise model is used.
    estimate_bias : bool, optional
        Whether the bias is included as an estimated state in filtering.
    """
    def __init__(self, axis: np.ndarray, efficiency: float, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        self.axis = normalize(axis)
        self.efficiency = efficiency
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=1, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean (noise- and bias-free) Sun sensor reading.

        The Sun direction is obtained from the orbital state as

        .. math::

            \hat{\mathbf{s}} =
            \frac{\mathbf{s} - \mathbf{r}}{\|\mathbf{s} - \mathbf{r}\|},

        where :math:`\mathbf{s}` is the Sun position and :math:`\mathbf{r}` is
        the spacecraft position. The raw cosine term is

        .. math:: c = \hat{\mathbf{a}}^\top \hat{\mathbf{s}}.

        The illuminated part is then

        .. math:: \text{illumination} = \max(c, 0),

        and the clean reading is

        .. math::

            y =
            \begin{cases}
                \eta \,\text{illumination}, & \text{if sunlit}, \\
                0, & \text{otherwise}.
            \end{cases}

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector.
        os : Orbital_State
            Orbital and environmental model that provides positions and
            lighting state (via :meth:`Orbital_State.is_sunlit`).

        Returns
        -------
        numpy.ndarray
            Clean Sun sensor measurement of shape ``(1,)``.
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
            return np.zeros_like(illumination)
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the Sun sensor bias.

        If a bias model is present, the measurement is modeled as

        .. math:: z = y + b,

        with scalar bias :math:`b`. Therefore,

        .. math:: \frac{\partial z}{\partial b} = 1.

        If no bias is present, an empty Jacobian is returned.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : Orbital_State
            Orbital state object (unused).

        Returns
        -------
        numpy.ndarray
            ``(1,1)`` matrix containing ``1`` if bias exists, otherwise a
            ``(0,1)`` empty matrix.
        """

        if self.bias:
            return np.ones((1,1))
        else:
            return np.zeros((0,1))
        
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean Sun sensor measurement with respect to the
        base attitude state

        .. math:: x = [\omega_x, \omega_y, \omega_z, q_0, q_1, q_2, q_3].

        Let

        .. math::

            \hat{\mathbf{s}} =
            \frac{\mathbf{s} - \mathbf{r}}{\|\mathbf{s} - \mathbf{r}\|},
            \qquad
            c = \hat{\mathbf{a}}^\top \hat{\mathbf{s}}.

        The derivative of the normalized Sun vector with respect to the state
        is provided by :func:`normed_vec_jac`, returning
        :math:`\partial \hat{\mathbf{s}} / \partial x`. The derivative of the
        cosine incidence term is

        .. math::

            \frac{\partial c}{\partial x}
            = \left( \frac{\partial \hat{\mathbf{s}}}{\partial x} \right)^\top
              \hat{\mathbf{a}}.

        Because the measurement uses ``max(c, 0)``, this Jacobian is zero
        whenever :math:`c \le 0`. When the spacecraft is in eclipse
        (``os.is_sunlit() == False``), the measurement is identically zero and
        the Jacobian is also zero.

        Parameters
        ----------
        x : numpy.ndarray
            Full 7-element ADCS state vector.
        os : Orbital_State
            Orbital state providing Sun/spacecraft geometry and lighting
            information.

        Returns
        -------
        numpy.ndarray
            A ``(7,1)`` Jacobian vector :math:`\partial y / \partial x` if
            illuminated, or zeros if in eclipse or if the Sun is behind the
            sensor (cosine term non-positive).
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
            return np.zeros((7, 1))
        
    

