__all__ = ["StarTracker"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_hardware.errors import Bias, AnisotropicNoise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import drotmatTvecdq, rot_mat
from ADCS.orbits.orbital_state import Orbital_State

class StarTracker(Sensor):
    r"""
    Star tracker direction sensor model.

    This class implements a simplified but estimator-consistent **star tracker**
    that provides a **vector observation** corresponding to the line-of-sight
    (LOS) from the spacecraft to a selected navigation star, expressed in the
    spacecraft body frame.

    Unlike a full star tracker attitude solution, this model returns a **single
    unit vector measurement**, making it suitable for tightly coupled state
    estimation frameworks such as EKF or UKF.

    The class conforms to the generic sensor interface defined by
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`.

    Measurement model
    -----------------
    Let

    ============================== ============================================
    Symbol                          Description
    ============================== ============================================
    :math:`\mathbf{q}`              Attitude quaternion (body → inertial)
    :math:`\mathbf{s}_{\mathrm{ECI}}`
                                    Inertial-frame unit vector to a star
    :math:`\mathbf{C}(\mathbf{q})`  Rotation matrix (body → inertial)
    ============================== ============================================

    The clean (ideal) star tracker measurement is

    .. math::

        \mathbf{y}_{\text{clean}}
        =
        \mathbf{C}(\mathbf{q})^\top \mathbf{s}_{\mathrm{ECI}}
        \in \mathbb{R}^3,

    which represents the star direction expressed in the **body frame**.

    Including bias and noise, the full measurement is

    .. math::

        \tilde{\mathbf{y}}
        =
        \mathbf{y}_{\text{clean}}
        + \mathbf{b}
        + \mathbf{n},

    where:

    * :math:`\mathbf{b}` is an optional additive bias modeled by
      :class:`~ADCS.satellite_hardware.errors.bias.Bias`
    * :math:`\mathbf{n}` is anisotropic noise modeled by
      :class:`~ADCS.satellite_hardware.errors.noise.AnisotropicNoise`

    After corruption, the measurement is **renormalized** to enforce a
    unit-vector constraint:

    .. math::

        \tilde{\mathbf{y}} \leftarrow
        \frac{\tilde{\mathbf{y}}}{\lVert \tilde{\mathbf{y}} \rVert}.

    Star selection
    --------------
    At each measurement epoch, the sensor:

    1. Projects the sensor boresight into the inertial frame
    2. Queries the :class:`~ADCS.environment.StarCatalog` for visible stars
    3. Applies field-of-view and exclusion constraints (Sun, optional Moon)
    4. Selects the **brightest visible star** (minimum visual magnitude)

    If no valid star is available, the sensor output is ``NaN``.

    Estimator properties
    --------------------
    * Output dimension: 3
    * Depends only on attitude quaternion
    * Jacobian is nonzero only w.r.t. quaternion states
    * No coupling to angular velocity or momentum states

    See Also
    --------
    :class:`~ADCS.environment.StarCatalog`  
    :class:`~ADCS.environment.NavigationStar`  
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`
    """

    output_length: int = 3

    def __init__(self, 
        sample_time: float = 0.1, 
        bias: Bias = None, 
        anisotropic_noise: AnisotropicNoise = None, 
        estimate_bias: bool = False,
        boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
        fov: float = np.deg2rad(4.0),
        sun_exclusion: float = np.deg2rad(25.0),
        star_catalog: Optional[StarCatalog] = None
    ) -> None:
        r"""
        Initialize the star tracker sensor.

        Parameters
        ----------
        sample_time : float
            Sampling period of the star tracker [s].
        bias : :class:`~ADCS.satellite_hardware.errors.bias.Bias`
            Optional additive bias model.
        anisotropic_noise : :class:`~ADCS.satellite_hardware.errors.noise.AnisotropicNoise`
            Direction-dependent noise model expressed in the sensor frame.
        estimate_bias : bool
            If ``True``, the bias is included in the estimator state.
        boresight : numpy.ndarray
            Sensor boresight direction in the body frame, shape ``(3,)``.
            This vector is normalized internally.
        fov : float
            Full-angle field of view of the star tracker [rad].
        sun_exclusion : float
            Minimum allowable angular separation between the boresight and the
            Sun direction [rad].
        star_catalog : :class:`~ADCS.environment.StarCatalog`
            Navigation star catalog used for visibility queries.

        :return: None
        :rtype: None
        """
        
        # 1. Geometry Setup
        self.boresight = np.asarray(boresight, dtype=np.float64)
        norm = np.linalg.norm(self.boresight)
        if norm < 1e-6:
            raise ValueError("Boresight vector cannot be zero.")
        self.boresight = self.boresight / norm

        self.fov = float(fov)
        self.sun_exclusion = float(sun_exclusion)
        
        self.catalog = star_catalog if star_catalog is not None else StarCatalog()
        self.current_star: Optional[NavigationStar] = None

        # 2. Calculate Rotation (Aligned -> Body)
        self._R_noise = self._build_noise_rotation()

        # 3. Initialize Base Sensor
        #    This will execute: self.noise = anisotropic_noise.copy()
        super().__init__(
            sample_time=sample_time,
            output_length=3,
            bias=bias,
            noise=anisotropic_noise,
            estimate_bias=estimate_bias
        )

        # 4. Align the (copied) noise model to this specific sensor's body frame
        if isinstance(self.noise, AnisotropicNoise):
            self.noise.align_to_body(self._R_noise)

    def _build_noise_rotation(self) -> NDArray[np.float64]:
        r"""
        Construct a rotation matrix that aligns the sensor boresight with the
        positive body-frame :math:`\hat{z}` axis.

        This rotation is used to express anisotropic noise statistics in the
        physical sensor frame.

        :return:
            Rotation matrix mapping the nominal sensor frame to the spacecraft
            body frame.
        :rtype: numpy.ndarray
        """
        z = np.array([0.0, 0.0, 1.0])

        if np.allclose(self.boresight, z):
            return np.eye(3)
        if np.allclose(self.boresight, -z):
            return np.diag([1.0, -1.0, -1.0])

        v = np.cross(z, self.boresight)
        s = np.linalg.norm(v)
        c = np.dot(z, self.boresight)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return R

    def _get_sun_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        if hasattr(os, 'S') and os.S is not None:
            s = np.asarray(os.S, dtype=np.float64)
            if not np.allclose(s, 0):
                return s
        return None

    def _get_moon_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        try:
            if hasattr(os, 'ephem') and os.ephem is not None:
                moon = os.ephem.planets['moon']
                moon_icrf = os.ephem.earth.at(os.sf_pos.t).observe(moon).apparent()
                return np.asarray(moon_icrf.position.km, dtype=np.float64)
        except (KeyError, AttributeError):
            pass
        return None

    def _select_star(self, q: NDArray[np.float64], os: Orbital_State) -> Optional[NavigationStar]:
        r"""
        Select the brightest visible navigation star.

        Visibility is determined using the
        :class:`~ADCS.environment.StarCatalog` based on:

        * Sensor boresight direction
        * Field-of-view constraint
        * Spacecraft position
        * Sun exclusion angle
        * Optional Moon exclusion

        Parameters
        ----------
        q : numpy.ndarray
            Spacecraft attitude quaternion (body → inertial), shape ``(4,)``.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing spacecraft position and ephemerides.

        :return:
            Brightest visible navigation star, or ``None`` if no valid star is
            available.
        :rtype: :class:`~ADCS.environment.NavigationStar` or None
        """

        A = rot_mat(q)
        boresight_eci = A @ self.boresight
        r_sat_eci = os.R
        sun_eci = self._get_sun_eci(os)
        moon_eci = self._get_moon_eci(os)

        visible = self.catalog.get_visible_stars(
            boresight_eci=boresight_eci,
            fov_rad=self.fov,
            r_sat_eci=r_sat_eci,
            sun_eci=sun_eci,
            moon_eci=moon_eci,
            sun_exclusion_rad=self.sun_exclusion
        )

        if not visible:
            return None

        return min(visible, key=lambda s: s.vmag)

    def clean_reading(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        r"""
        Compute the noise-free, bias-free star tracker measurement.

        The attitude quaternion is extracted from the state vector as

        .. math::

            \mathbf{q} = x_{3:7},

        and the measurement is computed as

        .. math::

            \mathbf{y}_{\text{clean}}
            =
            \mathbf{C}(\mathbf{q})^\top \mathbf{s}_{\mathrm{ECI}}.

        Parameters
        ----------
        x : numpy.ndarray
            Full spacecraft state vector.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state used for star visibility determination.

        :return:
            Unit vector pointing toward the selected navigation star in the
            body frame, or ``NaN`` if no star is visible.
        :rtype: numpy.ndarray
        """
        q = x[3:7]
        star = self._select_star(q, os)
        
        if star is None:
            self.current_star = None
            return np.full(3, np.nan)

        self.current_star = star
        A = rot_mat(q)
        return A.T @ star.s_eci

    def reading(self, x: NDArray[np.float64], os: Orbital_State, dmode: Optional[ErrorMode] = None) -> NDArray[np.float64]:
        r"""
        Compute the full star tracker measurement including bias and noise.

        Bias and noise are applied using
        :meth:`~ADCS.satellite_hardware.sensors.sensor.Sensor.reading`,
        after which the output is renormalized to enforce a unit-vector
        constraint.

        Parameters
        ----------
        x : numpy.ndarray
            Full spacecraft state vector.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state.
        dmode : :class:`~ADCS.satellite_hardware.errors.ErrorMode`
            Controls whether bias and noise are applied and propagated.

        :return:
            Normalized star direction measurement in the body frame.
        :rtype: numpy.ndarray
        """
        # Sensor.reading() handles clean + bias + noise
        measurement = super().reading(x, os, dmode)
        
        # Enforce unit vector constraint
        if not np.any(np.isnan(measurement)):
            norm = np.linalg.norm(measurement)
            if norm > 1e-9:
                measurement = measurement / norm
                
        return measurement

    def basestate_jac(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        r"""
        Jacobian of the star tracker measurement with respect to the base state.

        The measurement depends **only** on the attitude quaternion. Therefore,

        .. math::

            \frac{\partial \mathbf{y}}{\partial \boldsymbol{\omega}} = \mathbf{0},
            \qquad
            \frac{\partial \mathbf{y}}{\partial \mathbf{q}}
            =
            D_{\mathbf{q}}
            \left(
                \mathbf{C}(\mathbf{q})^\top \mathbf{s}_{\mathrm{ECI}}
            \right).

        The quaternion derivative is computed using
        :func:`~ADCS.helpers.math_helpers.drotmatTvecdq`.

        Parameters
        ----------
        x : numpy.ndarray
            Full spacecraft state vector.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state.

        :return:
            Base-state Jacobian stacked as ``[ω; q]``, shape ``(7, 3)``.
        :rtype: numpy.ndarray
        """
        if self.current_star is None:
            return np.zeros((7, self.output_length))

        q = x[3:7]
        s_eci = self.current_star.s_eci
        db_dq = drotmatTvecdq(q, s_eci)

        J = np.zeros((7, self.output_length))
        J[3:7, :] = db_dq
        return J

    def bias_jac(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        r"""
        Jacobian of the measurement with respect to bias states.

        The star tracker bias is not included in the estimator state for this
        model, so the bias Jacobian is empty.

        :return:
            Empty bias Jacobian of shape ``(0, 3)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((0, self.output_length))

    @property
    def noise_covariance(self) -> NDArray[np.float64]:
        r"""
        Measurement noise covariance matrix.

        :return:
            Measurement noise covariance expressed in the spacecraft body frame.
        :rtype: numpy.ndarray
        """
        if self.noise:
            return self.noise.cov()
        return np.zeros((3, 3))