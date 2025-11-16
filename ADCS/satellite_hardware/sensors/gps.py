__all__ = ["GPS"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.helpers.math_constants import MathConstants

class GPS(Sensor):
    r"""
    GPS position–velocity sensor model.

    This sensor provides a 6-element measurement consisting of

    * the satellite position in ECEF coordinates, and  
    * the satellite velocity expressed in ECEF coordinates.

    The *clean* measurement is constructed from the orbital state as

    .. math::

        \mathbf{z}
        =
        \begin{bmatrix}
            \mathbf{r}_{\mathrm{ECEF}} \\
            \mathbf{v}_{\mathrm{ECEF}}
        \end{bmatrix}
        \in \mathbb{R}^6.

    Parameters
    ----------
    sample_time : float, optional
        Sampling period of the GPS sensor in seconds (default: ``0.1``).
    bias : Bias, optional
        Bias model (6-element bias vector). If omitted, a zero-bias model is
        used.
    noise : Noise, optional
        Noise model (6-element noise vector). If omitted, a zero-noise model is
        used.
    estimate_bias : bool, optional
        If ``True``, the GPS bias is included as part of the estimated state
        in filtering algorithms. Otherwise it is treated as known or zero.
    """
    def __init__(self, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=6, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the clean GPS measurement (position + velocity in ECEF).

        This method uses the orbital state to extract

        * ``os.ECEF`` – satellite position expressed in ECEF coordinates, and
        * ``os.V`` – satellite velocity in ECI, which is transformed to ECEF
          using :meth:`Orbital_State.eci_to_ecef`.

        The returned measurement is

        .. math::

            \mathbf{z}_{\text{clean}}
            =
            \begin{bmatrix}
                \mathbf{r}_{\mathrm{ECEF}} \\
                \mathbf{v}_{\mathrm{ECEF}}
            \end{bmatrix}.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused for GPS computation).
        os : Orbital_State
            Orbital and environment model that provides position, velocity,
            and reference-frame transforms.

        Returns
        -------
        numpy.ndarray
            Clean 6-element GPS measurement ``[pos_ECEF, vel_ECEF]``.
        """
        ecef = os.ECEF
        v = os.V
        return np.concatenate([ecef, os.eci_to_ecef(v)])
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the GPS bias state.

        The GPS measurement is modeled as

        .. math:: \mathbf{z} = \mathbf{z}_{\text{clean}} + \mathbf{b},

        where :math:`\mathbf{b} \in \mathbb{R}^6` is a 6-element bias vector.
        Therefore,

        .. math::

            \frac{\partial \mathbf{z}}{\partial \mathbf{b}}
            = I_6.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state (unused).
        os : Orbital_State
            Orbital state object (unused).

        Returns
        -------
        numpy.ndarray
            ``6 × 6`` identity matrix if a bias model exists,
            otherwise a ``0 × 6`` empty matrix.
        """
        if self.bias:
            return np.eye(6)
        else:
            return np.zeros((0, 6))
        
    def orbitRV_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the GPS measurement with respect to the orbital position
        and velocity in the ECI frame.

        The position and velocity transformations are:

        * Position: :math:`\mathbf{r}_{\mathrm{ECEF}} = \mathbf{r}_{\mathrm{ECEF}}(\mathbf{r}_{\mathrm{ECI}})`
        * Velocity: :math:`\mathbf{v}_{\mathrm{ECEF}} = C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} \, \mathbf{v}_{\mathrm{ECI}}`

        The Jacobian therefore consists of two identical blocks:

        .. math::

            \frac{\partial \mathbf{z}}{\partial [\mathbf{r}_{\mathrm{ECI}},\mathbf{v}_{\mathrm{ECI}}]}
            =
            \begin{bmatrix}
                C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}} & 0 \\
                0 & C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}}
            \end{bmatrix},

        where :math:`C_{\mathrm{ECI}\rightarrow\mathrm{ECEF}}` is the
        coordinate-frame rotation matrix.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : Orbital_State
            Orbital state containing the frame-transformation function
            :meth:`eci_to_ecef`.
        Returns
        -------
        numpy.ndarray
            A ``6 × 6`` block-diagonal Jacobian matrix.
        """
        mat = np.stack([os.eci_to_ecef(j) for j in MathConstants.unitvecs]).T
        return block_diag(mat, mat)

