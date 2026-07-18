__all__ = ["Sensor"]

import numpy as np

from ADCS.state import State

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.satellite_hardware.errors import ErrorMode

class Sensor:
    r"""
    Base class for all Attitude Determination and Control System (ADCS) sensor models.

    This class defines a **unified measurement interface** for all onboard sensors
    (e.g., gyroscopes, magnetometers, sun sensors, star trackers). It encapsulates
    the full sensor measurement pipeline, including:

    * Ideal (noise– and bias–free) measurement generation
    * Additive stochastic bias modeling
    * Additive measurement noise modeling
    * Time propagation of bias and noise processes
    * Analytical Jacobians for use in estimation filters (e.g., EKF, UKF)

    Subclasses must implement at least :meth:`clean_reading` and may override
    Jacobian methods as required by the sensor physics.

    Mathematical measurement model
    --------------------------------
    The generic sensor model implemented by this class is

    .. math::

        \mathbf{z}_k = \mathbf{h}(\mathbf{x}_k, \mathcal{O}_k)
        + \mathbf{b}_k + \mathbf{n}_k

    where:

    .. list-table:: Notation
        :widths: 30 70
        :header-rows: 1

        * - Symbol
          - Description
        * - :math:`\mathbf{z}_k`
          - Sensor measurement vector
        * - :math:`\mathbf{x}_k`
          - Full system state vector
        * - :math:`\mathcal{O}_k`
          - Orbital/environmental state
        * - :math:`\mathbf{h}`
          - Ideal (clean) sensor model
        * - :math:`\mathbf{b}_k`
          - Sensor bias
        * - :math:`\mathbf{n}_k`
          - Measurement noise


    Bias and noise are modeled using instances of
    :class:`~ADCS.satellite_hardware.errors.bias.Bias` and
    :class:`~ADCS.satellite_hardware.errors.noise.Noise`, respectively.

    Bias evolution is typically time-dependent:

    .. math::

        \mathbf{b}_{k+1} = f_b(\mathbf{b}_k, t_k)

    while noise is assumed to be white or colored stochastic noise:

    .. math::

        \mathbf{n}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})

    :param sample_time: Sampling period of the sensor in seconds.
    :type sample_time: float
    :param output_length: Dimension of the sensor measurement output.
    :type output_length: int
    :param bias: Bias model instance. If ``None``, a zero-bias model is used.
    :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
    :param noise: Noise model instance. If ``None``, a zero-noise model is used.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
    :param estimate_bias: Indicates whether the sensor bias is included in the estimator state.
    :type estimate_bias: bool
    :return: None
    :rtype: None

    Notes
    -----
    * This class does **not** assume any specific state ordering.
    * Bias and noise updates are controlled via
      :class:`~ADCS.satellite_hardware.errors.ErrorMode`.
    """
    def __init__(self, sample_time: float = 0.1, output_length: int = 1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        r"""
        Initialize a generic sensor model.

        :param sample_time: Sampling period of the sensor in seconds.
        :type sample_time: float
        :param output_length: Dimension of the sensor measurement output.
        :type output_length: int
        :param bias: Bias model instance. If ``None``, a zero-bias model is used.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Noise model instance. If ``None``, a zero-noise model is used.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: Indicates whether the sensor bias is included in the estimator state.
        :type estimate_bias: bool
        :return: None
        :rtype: None
        """
        if bias:
            self.bias = bias.copy()
        else:
            self.bias = Bias()
        if noise:
            self.noise = noise.copy()
        else:
            self.noise = Noise()
        self.sample_time = sample_time
        self.output_length = output_length
        self.estimate_bias = estimate_bias

    def reading(self, x: State, os: Orbital_State, dmode: ErrorMode = None) -> np.ndarray:
        r"""
        Compute the full sensor measurement including bias and noise.

        This method evaluates the complete sensor pipeline:

        1. Compute the clean (ideal) sensor reading
        2. Add the current sensor bias
        3. Update the bias process (if enabled)
        4. Add measurement noise
        5. Update the noise process (if enabled)

        Formally, the measurement is computed as

        .. math::

            \mathbf{z} =
            \mathbf{h}(\mathbf{x}, \mathcal{O})
            + \mathbf{b}(t)
            + \mathbf{n}

        where:

        * :math:`\mathbf{h}` is implemented by :meth:`clean_reading`
        * :math:`\mathbf{b}(t)` is provided by
          :class:`~ADCS.satellite_hardware.errors.bias.Bias`
        * :math:`\mathbf{n}` is provided by
          :class:`~ADCS.satellite_hardware.errors.noise.Noise`

        The inclusion and propagation of bias and noise are controlled via
        :class:`~ADCS.satellite_hardware.errors.ErrorMode`.

        :param x: Full system state vector.
        :type x: ADCS.state.State
        :param os: Orbital and environmental state providing the current time ``os.J2000``.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param dmode: Error mode configuration controlling bias and noise behavior.
                      If ``None``, all effects are enabled.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` or None
        :return: Sensor measurement vector.
        :rtype: numpy.ndarray

        """
        if dmode is None:
            dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        reading = self.clean_reading(x=x, os=os)
        # Evolve the stochastic states FIRST so this reading uses the noise/bias
        # realization for the current step. Previously get_noise()/get_bias()
        # were added before _update_*(), so every sensor's first measurement
        # used the initial (zero) noise sample -- i.e. the first reading was
        # noise-free and bias diffusion lagged by one step.
        if dmode.update_bias:
            self.bias._update_bias(os.J2000)
        if self.bias and dmode.add_bias:
            reading += self.bias.get_bias(os.J2000)

        if dmode.update_noise:
            self.noise._update_noise()
        if self.noise and dmode.add_noise:
            reading += self.noise.get_noise()

        return reading
    
    def basestate_jac(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the clean sensor measurement with respect to the base states.

        This method returns the Jacobian

        .. math::

            \mathbf{H}_x =
            \frac{\partial \mathbf{h}(\mathbf{x}, \mathcal{O})}
            {\partial \mathbf{x}_{\text{base}}}

        where :math:`\mathbf{x}_{\text{base}}` denotes the non-bias portion of the
        system state.

        The default implementation assumes that the measurement is **independent**
        of the base state and returns a zero matrix. Sensor subclasses should
        override this method to provide physically meaningful derivatives.

        :param x: Full system state vector.
        :type x: ADCS.state.State
        :param os: Orbital and environmental state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Jacobian matrix of shape ``(output_length, N_base_states)``.
        :rtype: numpy.ndarray

        """
        return np.zeros((7, self.output_length))
    
    def bias_jac(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the measurement with respect to the sensor bias state.

        If the measurement model is

        .. math::

            \mathbf{z} = \mathbf{h}(\mathbf{x}, \mathcal{O}) + \mathbf{b},

        then the Jacobian with respect to the bias is

        .. math::

            \mathbf{H}_b =
            \frac{\partial \mathbf{z}}{\partial \mathbf{b}}
            = \mathbf{I}

        where :math:`\mathbf{I}` is the identity matrix of dimension
        ``output_length``.

        If the sensor does not include a bias model, an empty Jacobian is returned.

        :param x: Full system state vector (unused).
        :type x: ADCS.state.State
        :param os: Orbital state object (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Identity matrix of shape ``(output_length, output_length)`` if a bias
                 model exists; otherwise an empty matrix.
        :rtype: numpy.ndarray

        """
        if self.bias:
            return np.eye(self.output_length)
        else:
            return np.zeros((0, self.output_length))