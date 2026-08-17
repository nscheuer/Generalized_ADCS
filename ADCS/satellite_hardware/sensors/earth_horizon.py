__all__ = ["EarthHorizonSensor"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import rot_mat, drotmatTvecdq
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants


class EarthHorizonSensor(Sensor):
    r"""
    Earth horizon crossing sensor model.

    This class implements a **scanning horizon sensor** that determines the
    spacecraft nadir direction by detecting transitions across the Earth's
    limb.  The sensor output is a unit vector pointing toward the Earth
    center expressed in the spacecraft body frame.

    Physical principle
    ------------------
    A horizon scanner sweeps a narrow beam across the Earth disk and records
    the angles at which infrared radiance transitions occur (bright Earth to
    cold space or vice versa).  From the geometry of these limb crossings,
    the nadir direction is reconstructed.

    In this simplified model, the reconstructed nadir direction is assumed
    to be the true nadir direction plus additive bias and noise, consistent
    with the generic sensor pipeline defined by
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`.

    Measurement model
    -----------------
    Let

    ================================= ===========================================
    Symbol                            Description
    ================================= ===========================================
    :math:`\mathbf{q}`                Attitude quaternion (body -> inertial)
    :math:`\mathbf{R}_{\mathrm{sat}}` Spacecraft position in ECI [km]
    :math:`\mathbf{C}(\mathbf{q})`    Rotation matrix (body -> inertial)
    :math:`R_\oplus`                  Mean Earth radius (6378.137 km)
    ================================= ===========================================

    The nadir direction in the ECI frame is

    .. math::

        \hat{\mathbf{n}}_{\mathrm{ECI}}
        = -\frac{\mathbf{R}_{\mathrm{sat}}}
               {\|\mathbf{R}_{\mathrm{sat}}\|}

    and the clean measurement is the nadir direction in the body frame:

    .. math::

        \mathbf{y}_{\text{clean}}
        = \mathbf{C}(\mathbf{q})^\top\,\hat{\mathbf{n}}_{\mathrm{ECI}}

    The Earth subtends an angular radius

    .. math::

        \rho = \arcsin\!\left(\frac{R_\oplus}{\|\mathbf{R}_{\mathrm{sat}}\|}\right)

    which determines the measurement accuracy achievable by a real horizon
    scanner.

    Visibility constraint
    ---------------------
    The sensor returns a valid measurement only when the nadir direction
    falls within the sensor field of view:

    .. math::

        \arccos\!\bigl(\hat{\mathbf{b}} \cdot \mathbf{y}_{\text{clean}}\bigr)
        < \theta_{\mathrm{FOV}}

    where :math:`\hat{\mathbf{b}}` is the sensor boresight in the body frame
    and :math:`\theta_{\mathrm{FOV}}` is the half-cone field of view.
    If this condition is not met, the sensor output is ``NaN``.

    After adding bias and noise, the output is renormalized to enforce the
    unit-vector constraint.

    Estimator properties
    --------------------
    * Output dimension: 3
    * Depends on attitude quaternion (for frame rotation)
    * Uses orbital position for nadir direction (treated as known)
    * Jacobian is nonzero only w.r.t. quaternion states
    * No coupling to angular velocity or momentum states

    See Also
    --------
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`
    :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`
    """

    output_length: int = 3

    def __init__(
        self,
        sample_time: float = 0.1,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
        boresight: np.ndarray = np.array([0.0, 0.0, -1.0]),
        fov: float = np.deg2rad(90.0),
    ) -> None:
        r"""
        Initialize the Earth horizon sensor.

        :param sample_time: Sampling period [s].
        :type sample_time: float
        :param bias: Optional additive bias model (3-element).
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Optional noise model (3-element).
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: If ``True``, bias is included in estimator state.
        :type estimate_bias: bool
        :param boresight: Sensor boresight direction in body frame, shape ``(3,)``.
                          Default is ``[0, 0, -1]`` (nadir-pointing).
                          Normalized internally.
        :type boresight: numpy.ndarray
        :param fov: Half-cone field of view [rad].  The sensor returns a valid
                    reading only if nadir falls within this cone around the
                    boresight.  Default is 90 deg (hemisphere).
        :type fov: float
        :return: None
        :rtype: None
        """
        self.boresight = np.asarray(boresight, dtype=np.float64)
        norm = np.linalg.norm(self.boresight)
        if norm < 1e-6:
            raise ValueError("Boresight vector cannot be zero.")
        self.boresight = self.boresight / norm

        self.fov = float(fov)
        self._R_earth = EarthConstants.R_e  # km

        # Cache for Jacobian computation
        self._nadir_eci: Optional[NDArray[np.float64]] = None

        super().__init__(
            sample_time=sample_time,
            output_length=3,
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias,
        )

    @property
    def earth_angular_radius(self) -> Optional[float]:
        r"""
        Earth angular radius from the last measurement [rad].

        Computed as :math:`\rho = \arcsin(R_\oplus / \|\mathbf{r}\|)`.
        Returns ``None`` if no measurement has been taken.

        :rtype: float or None
        """
        return self._earth_angular_radius if hasattr(self, "_earth_angular_radius") else None

    def clean_reading(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Compute the noise-free nadir direction measurement.

        The nadir direction is computed from the orbital position and
        rotated into the body frame using the attitude quaternion:

        .. math::

            \mathbf{y}_{\text{clean}}
            = \mathbf{C}(\mathbf{q})^\top
              \left(-\frac{\mathbf{R}_{\mathrm{sat}}}{\|\mathbf{R}_{\mathrm{sat}}\|}\right)

        A visibility check ensures the nadir direction falls within the
        sensor field of view.

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state providing spacecraft position.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Nadir unit vector in body frame, or ``NaN`` if Earth is
                 outside the sensor FOV.
        :rtype: numpy.ndarray
        """
        q = x.q
        r_sat = os.R  # ECI position [km]
        r_norm = np.linalg.norm(r_sat)

        # Nadir direction in ECI
        nadir_eci = -r_sat / r_norm
        self._nadir_eci = nadir_eci

        # Earth angular radius
        self._earth_angular_radius = np.arcsin(
            np.clip(self._R_earth / r_norm, -1.0, 1.0)
        )

        # Rotate nadir to body frame
        A = rot_mat(q)
        nadir_body = A.T @ nadir_eci

        # FOV visibility check
        cos_angle = np.dot(self.boresight, nadir_body)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        if angle > self.fov:
            self._nadir_eci = None
            return np.full(3, np.nan)

        return nadir_body

    def reading(
        self,
        x: NDArray[np.float64],
        os: Orbital_State,
        dmode: Optional[ErrorMode] = None,
    ) -> NDArray[np.float64]:
        r"""
        Compute the full nadir measurement including bias and noise.

        After the base class applies bias and noise, the output is
        renormalized to enforce the unit-vector constraint.

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param dmode: Error mode controlling bias and noise.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` or None
        :return: Normalized nadir direction in body frame.
        :rtype: numpy.ndarray
        """
        measurement = super().reading(x, os, dmode)

        if not np.any(np.isnan(measurement)):
            norm = np.linalg.norm(measurement)
            if norm > 1e-9:
                measurement = measurement / norm

        return measurement

    def basestate_jac(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Jacobian of the nadir measurement w.r.t. the base state.

        The measurement depends only on the attitude quaternion (the orbital
        position is treated as known). Therefore:

        .. math::

            \frac{\partial \mathbf{y}}{\partial \boldsymbol{\omega}}
            = \mathbf{0},
            \qquad
            \frac{\partial \mathbf{y}}{\partial \mathbf{q}}
            = D_{\mathbf{q}}
            \left(
                \mathbf{C}(\mathbf{q})^\top\,\hat{\mathbf{n}}_{\mathrm{ECI}}
            \right)

        The quaternion derivative is computed using
        :func:`~ADCS.helpers.math_helpers.drotmatTvecdq`.

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Base-state Jacobian of shape ``(7, 3)``.
        :rtype: numpy.ndarray
        """
        if self._nadir_eci is None:
            return np.zeros((7, self.output_length))

        q = x.q
        db_dq = drotmatTvecdq(q, self._nadir_eci)

        J = np.zeros((7, self.output_length))
        J[3:7, :] = db_dq
        return J

    def bias_jac(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Jacobian of the measurement w.r.t. bias states.

        If ``estimate_bias`` is ``True``, the bias Jacobian is the identity.
        Otherwise, an empty Jacobian is returned.

        :param x: Full spacecraft state vector (unused).
        :type x: ADCS.state.State
        :param os: Orbital state (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Bias Jacobian.
        :rtype: numpy.ndarray
        """
        if self.estimate_bias:
            return np.eye(self.output_length)
        return np.zeros((0, self.output_length))
