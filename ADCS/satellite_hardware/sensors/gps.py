__all__ = ["GPS"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_constants import MathConstants

class GPS(Sensor):
    r"""
    Global Positioning System (GPS) position–velocity sensor model.

    This class implements a GPS receiver that provides a **6-element measurement**
    consisting of the spacecraft position and velocity expressed in the
    Earth-Centered Earth-Fixed (ECEF) reference frame.

    The GPS sensor follows the generic sensor model defined in
    :class:`~ADCS.satellite_hardware.sensor.Sensor`.

    Measurement model
    ------------------
    The clean (ideal) GPS measurement is defined as

    .. math::

        \mathbf{z}_{\text{clean}}
        =
        \begin{bmatrix}
            \mathbf{r}_{\mathrm{ECEF}} \\
            \mathbf{v}_{\mathrm{ECEF}}
        \end{bmatrix}
        \in \mathbb{R}^6,

    where:

    ======================= ============================================
    Symbol                  Description
    ======================= ============================================
    :math:`\mathbf{r}`      Satellite position vector
    :math:`\mathbf{v}`      Satellite velocity vector
    ECEF                    Earth-Centered Earth-Fixed frame
    ======================= ============================================

    The full measurement includes bias and noise:

    .. math::

        \mathbf{z}
        =
        \mathbf{z}_{\text{clean}}
        + \mathbf{b}
        + \mathbf{n},

    where:

    * :math:`\mathbf{b}` is a 6-element GPS bias vector modeled using
      :class:`~ADCS.satellite_hardware.errors.Bias`
    * :math:`\mathbf{n}` is a 6-element measurement noise vector modeled using
      :class:`~ADCS.satellite_hardware.errors.Noise`

    Parameters
    ----------
    sample_time : float
        Sampling period of the GPS sensor in seconds.
    bias : :class:`~ADCS.satellite_hardware.errors.Bias`
        Bias model for the GPS measurement. Must be 6-dimensional.
        If ``None``, a zero-bias model is used.
    noise : :class:`~ADCS.satellite_hardware.errors.Noise`
        Noise model for the GPS measurement. Must be 6-dimensional.
        If ``None``, a zero-noise model is used.
    estimate_bias : bool
        Flag indicating whether the GPS bias is included as part of the
        estimated filter state.

    Notes
    -----
    * This sensor is **not** an attitude sensor.
    * Position is taken directly from the orbital state in ECEF coordinates.
    * Velocity is transformed from ECI to ECEF using
      :meth:`~ADCS.orbits.orbital_state.Orbital_State.eci_to_ecef`.
    """
    def __init__(self, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        r"""
        Initialize the GPS sensor model.

        Parameters
        ----------
        sample_time : float
            Sampling period of the GPS sensor in seconds.
        bias : :class:`~ADCS.satellite_hardware.errors.Bias` or None
            GPS bias model. If ``None``, a zero 6-element bias is used.
        noise : :class:`~ADCS.satellite_hardware.errors.Noise` or None
            GPS noise model. If ``None``, a zero 6-element noise model is used.
        estimate_bias : bool
            Indicates whether the GPS bias is estimated as part of the system
            state.

        :return: None
        :rtype: None
        """
        self.attitude_sensor = False
        if bias:
            self.bias = bias
        else:
            self.bias = Bias(bias=np.zeros(6), std_bias=np.zeros(6))
        if noise:
            self.noise = noise
        else:
            self.noise = Noise(noise=np.zeros(6), std_noise=np.zeros(6))
        super().__init__(sample_time=sample_time, output_length=6, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean GPS measurement (position and velocity in ECEF).

        This method constructs the ideal GPS output using the orbital state:

        * Position is obtained directly from ``os.ECEF``
        * Velocity is obtained from ``os.V`` (ECI) and rotated into ECEF using
          :meth:`~ADCS.orbits.orbital_state.Orbital_State.eci_to_ecef`

        Mathematically,

        .. math::

            \mathbf{z}_{\text{clean}}
            =
            \begin{bmatrix}
                \mathbf{r}_{\mathrm{ECEF}} \\
                C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} \, \mathbf{v}_{\mathrm{ECI}}
            \end{bmatrix},

        where :math:`C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}}` is the coordinate
        transformation matrix from the inertial to Earth-fixed frame.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector. This argument is unused by the GPS model.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing position, velocity, and frame transforms.

        :return:
            Clean GPS measurement vector
            ``[r_ECEF, v_ECEF]``.
        :rtype: numpy.ndarray
        """
        ecef = os.ECEF
        v = os.V
        return np.concatenate([ecef, os.eci_to_ecef(v)])
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the GPS measurement with respect to the GPS bias state.

        The GPS measurement is modeled as

        .. math::

            \mathbf{z}
            =
            \mathbf{z}_{\text{clean}} + \mathbf{b},

        where :math:`\mathbf{b} \in \mathbb{R}^6` is the GPS bias vector.

        Therefore, the Jacobian with respect to the bias is

        .. math::

            \mathbf{H}_b
            =
            \frac{\partial \mathbf{z}}{\partial \mathbf{b}}
            =
            I_6.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state object (unused).

        :return:
            ``6 × 6`` identity matrix if a bias model exists;
            otherwise an empty ``0 × 6`` matrix.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return np.eye(6)
        else:
            return np.zeros((0, 6))
        
    def orbitRV_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the GPS measurement with respect to orbital position and
        velocity states in the ECI frame.

        The coordinate transformations are:

        * Position:
          :math:`\mathbf{r}_{\mathrm{ECEF}}
          = C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} \mathbf{r}_{\mathrm{ECI}}`
        * Velocity:
          :math:`\mathbf{v}_{\mathrm{ECEF}}
          = C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} \mathbf{v}_{\mathrm{ECI}}`

        The resulting Jacobian is block-diagonal:

        .. math::

            \mathbf{H}_{rv}
            =
            \frac{\partial \mathbf{z}}
            {\partial
            \begin{bmatrix}
                \mathbf{r}_{\mathrm{ECI}} \\
                \mathbf{v}_{\mathrm{ECI}}
            \end{bmatrix}}
            =
            \begin{bmatrix}
                C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} & 0 \\
                0 & C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}}
            \end{bmatrix}.

        The rotation matrix is constructed by applying
        :meth:`~ADCS.orbits.orbital_state.Orbital_State.eci_to_ecef` to the unit
        basis vectors defined in
        :class:`~ADCS.helpers.math_constants.MathConstants`.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing the ECI→ECEF transformation.

        :return:
            Block-diagonal Jacobian matrix of shape ``6 × 6``.
        :rtype: numpy.ndarray
        """
        mat = np.stack([os.eci_to_ecef(j) for j in MathConstants.unitvecs]).T
        return block_diag(mat, mat)

