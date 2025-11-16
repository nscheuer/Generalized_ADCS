__all__ = ["Sensor"]

import numpy as np

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias

class Sensor:
    r"""
    Base class for all ADCS sensor models.

    This class defines a standard interface for sensor measurement generation,
    including clean (noise– and bias–free) readings, application of stochastic
    bias and noise models, and measurement Jacobians for filtering (e.g., EKF).

    Concrete sensor subclasses (e.g., :class:`Gyro`, :class:`MTM`,
    star trackers, sun sensors) must implement at least
    :meth:`clean_reading` and may override Jacobian functions as needed.

    Parameters
    ----------
    sample_time : float, optional
        Sampling period of the sensor in seconds (default: ``0.1``).
    output_length : int, optional
        Dimension of the sensor measurement output (default: ``1``).
    bias : Bias, optional
        Bias model that provides additive measurement bias and bias evolution.
        If not provided, initializes with a zero bias model.
    noise : Noise, optional
        Noise model providing measurement noise with a chosen distribution.
        If not provided, initializes with a zero-noise model.
    estimate_bias : bool, optional
        Indicates whether the filter should include bias as part of the
        estimated state. (Stored by subclasses; not used directly here.)
    """
    def __init__(self, sample_time: float = 0.1, output_length: int = 1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        if bias:
            self.bias = bias
        else:
            self.bias = Bias(bias=np.zeros(6), std_bias=np.zeros(6))
        if noise:
            self.noise = noise
        else:
            self.noise = Noise(noise=np.zeros(6), std_noise=np.zeros(6))
        self.sample_time = sample_time
        self.output_length = output_length

    def reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the full sensor measurement, including bias and noise.

        The process is:

        1. Compute the clean measurement using :meth:`clean_reading`.
        2. Add the current sensor bias, obtained via :meth:`Bias.get_bias`.
        3. Add measurement noise, via :meth:`Noise.get_noise`.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector.
        os : Orbital_State
            Orbital and environmental model. Provides the time (``os.J2000``)
            required for time-varying bias processes.

        Returns
        -------
        numpy.ndarray
            The final sensor measurement, shape ``(output_length,)``.
        """
        reading = self.clean_reading(x=x, os=os)
        if self.bias:
            reading += self.bias.get_bias(os.J2000)

        if self.noise:
            reading += self.noise.get_noise()

        return reading
    
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean sensor measurement with respect to the
        base (non-bias) states.

        This base implementation returns a zero matrix, corresponding to a
        measurement independent of the satellite state. Subclasses should
        override this method to provide the correct Jacobian.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector.
        os : Orbital_State
            Orbital and environmental model.

        Returns
        -------
        numpy.ndarray
            Zero matrix of shape ``(7, output_length)`` by default.
        """
        return np.zeros((7, self.output_length))
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the sensor bias state.

        If the sensor has a bias model, the measurement is assumed to be

        .. math::

            z = y + b,

        where :math:`b` is a vector of length ``output_length``. Therefore,

        .. math::

            \frac{\partial z}{\partial b} = I,

        the identity matrix. If no bias model is present, an empty Jacobian is
        returned.

        Parameters
        ----------
        x : numpy.ndarray
            Full system state vector (unused).
        os : Orbital_State
            Orbital state object (unused).

        Returns
        -------
        numpy.ndarray
            ``(output_length, output_length)`` identity matrix if bias exists,
            otherwise a ``(0, output_length)`` empty matrix.
        """
        if self.bias:
            return np.eye(self.output_length)
        else:
            return np.zeros((0, self.output_length))