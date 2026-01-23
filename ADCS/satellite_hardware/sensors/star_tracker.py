__all__ = ["StarTracker"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_hardware.errors import Bias, AnisotropicNoise
from ADCS.satellite_hardware.disturbances.helpers.disturbance_mode import DisturbanceMode
from ADCS.helpers.math_helpers import drotmatTvecdq, rot_mat
from ADCS.orbits.orbital_state import Orbital_State

class StarTracker(Sensor):
    r"""
    **Star Tracker Sensor Model**

    This class implements a simplified but estimator-consistent **star tracker sensor**
    that measures the direction to a single navigation star expressed in the spacecraft
    body frame.

    The sensor outputs a **unit vector** corresponding to the line-of-sight (LOS)
    from the spacecraft to the brightest visible star within the field of view (FOV),
    subject to geometric visibility and exclusion constraints.

    The star tracker is modeled as a **direction sensor**, not a full attitude solver.
    It provides a vector observation suitable for use in EKF/UKF-style estimators.

    ---
    **Measurement Model**

    Let

    - :math:`\mathbf{q}` — spacecraft attitude quaternion (body → inertial)
    - :math:`\mathbf{s}_\mathrm{ECI}` — inertial-frame unit vector toward a navigation star
    - :math:`\mathbf{C}(\mathbf{q})` — rotation matrix mapping body → inertial

    The ideal (noise-free, bias-free) measurement is

    .. math::

        \mathbf{y}
        \;=\;
        \mathbf{C}(\mathbf{q})^\top \, \mathbf{s}_\mathrm{ECI}
        \;\in\; \mathbb{R}^3

    i.e. the star direction expressed in the **body frame**.

    ---
    **Bias and Noise**

    Measurement corruption is handled by the base class
    :class:`~ADCS.satellite_hardware.sensors.sensor.Sensor`:

    .. math::

        \tilde{\mathbf{y}} = \mathbf{y} + \mathbf{b} + \mathbf{n}

    where

    - :math:`\mathbf{b}` is an optional additive bias modeled by
      :class:`~ADCS.satellite_hardware.errors.bias.Bias`
    - :math:`\mathbf{n}` is optional anisotropic noise modeled by
      :class:`~ADCS.satellite_hardware.errors.noise.AnisotropicNoise`

    The final output is **renormalized** to enforce a unit-vector constraint.

    ---
    **Star Selection Logic**

    At each measurement time step, the sensor:

    1. Projects the sensor boresight into the inertial frame
    2. Queries the :class:`~ADCS.environment.StarCatalog` for visible stars
    3. Applies field-of-view, Sun exclusion, and optional Moon exclusion checks
    4. Selects the **brightest visible star** (minimum visual magnitude)

    If no valid star is available, the sensor returns ``NaN``.

    ---
    **Estimator Properties**

    - Output dimension: 3
    - Depends on attitude quaternion only
    - Jacobian is nonzero only w.r.t. quaternion states
    - No momentum or bias-state coupling

    This makes the model well-suited for tightly coupled attitude estimators.

    See Also
    --------
    ~ADCS.environment.StarCatalog  
    ~ADCS.environment.NavigationStar  
    ~ADCS.satellite_hardware.sensors.sensor.Sensor
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
        Initialize a star tracker sensor instance.

        Parameters
        ----------
        sample_time : float, optional
            Sampling period of the sensor [s].

        bias : ~ADCS.satellite_hardware.errors.bias.Bias, optional
            Additive measurement bias model.

        anisotropic_noise : ~ADCS.satellite_hardware.errors.noise.AnisotropicNoise, optional
            Direction-dependent noise model expressed in the sensor frame.

        estimate_bias : bool, optional
            If ``True``, the bias is included in the estimator state vector.

        boresight : ndarray, shape (3,), optional
            Sensor boresight direction expressed in the spacecraft body frame.
            This vector is normalized internally.

        fov : float, optional
            Full-angle field of view of the star tracker [rad].

        sun_exclusion : float, optional
            Minimum allowable angular separation between the sensor boresight
            and the Sun direction [rad].

        star_catalog : ~ADCS.environment.StarCatalog, optional
            Catalog of navigation stars used for visibility queries.
            If not provided, a default catalog is constructed.

        Notes
        -----
        The anisotropic noise covariance is rotated internally such that its
        principal axes are aligned with the sensor boresight.
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
        Construct a rotation matrix that aligns the sensor boresight
        with the positive body-frame :math:`\hat{z}` axis.

        This rotation is used to express anisotropic noise statistics
        in the physical sensor frame.

        Returns
        -------
        ndarray, shape (3, 3)
            Rotation matrix mapping the nominal sensor frame
            to the spacecraft body frame.
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
        :class:`~ADCS.environment.StarCatalog`, based on:

        - Sensor boresight direction
        - Field-of-view constraint
        - Spacecraft position
        - Sun exclusion angle
        - Optional Moon exclusion

        Parameters
        ----------
        q : ndarray, shape (4,)
            Spacecraft attitude quaternion (body → inertial).

        os : ~ADCS.orbits.orbital_state.Orbital_State
            Orbital state providing spacecraft position and ephemerides.

        Returns
        -------
        ~ADCS.environment.NavigationStar or None
            Brightest visible star, or ``None`` if no valid star is available.
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
        Compute the **noise-free, bias-free** star tracker measurement.

        Parameters
        ----------
        x : ndarray
            Full spacecraft state vector. The quaternion is extracted
            from ``x[3:7]``.

        os : ~ADCS.orbits.orbital_state.Orbital_State
            Orbital state used for star visibility determination.

        Returns
        -------
        ndarray, shape (3,)
            Unit vector pointing toward the selected navigation star
            expressed in the spacecraft body frame.
            Returns ``NaN`` if no star is visible.
        """
        q = x[3:7]
        star = self._select_star(q, os)
        
        if star is None:
            self.current_star = None
            return np.full(3, np.nan)

        self.current_star = star
        A = rot_mat(q)
        return A.T @ star.s_eci

    def reading(self, x: NDArray[np.float64], os: Orbital_State, dmode: Optional[DisturbanceMode] = None) -> NDArray[np.float64]:
        r"""
        Compute the full star tracker measurement including bias and noise.

        This method delegates bias and noise injection to
        :meth:`~ADCS.satellite_hardware.sensors.sensor.Sensor.reading`
        and then enforces a unit-vector constraint.

        Parameters
        ----------
        x : ndarray
            Full spacecraft state vector.

        os : ~ADCS.orbits.orbital_state.Orbital_State
            Orbital state.

        dmode : ~ADCS.satellite_hardware.disturbances.disturbance_mode.DisturbanceMode, optional
            Controls whether bias and noise are applied and updated.

        Returns
        -------
        ndarray, shape (3,)
            Normalized star direction measurement in the body frame.
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
        Compute the Jacobian of the star tracker measurement
        with respect to the spacecraft base state.

        The measurement depends **only on the attitude quaternion**.
        All derivatives with respect to angular velocity are zero.

        Let

        .. math::

            \mathbf{y} = \mathbf{C}(\mathbf{q})^\top \mathbf{s}_\mathrm{ECI}

        Then

        .. math::

            \frac{\partial \mathbf{y}}{\partial \boldsymbol{\omega}} = \mathbf{0}, \qquad
            \frac{\partial \mathbf{y}}{\partial \mathbf{q}}
            = D_\mathbf{q}\!\left(\mathbf{C}^\top \mathbf{s}_\mathrm{ECI}\right)

        Parameters
        ----------
        x : ndarray
            Spacecraft state vector.

        os : ~ADCS.orbits.orbital_state.Orbital_State
            Orbital state.

        Returns
        -------
        ndarray, shape (7, 3)
            Base-state Jacobian stacked as ``[ω; q]``.
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
        Jacobian of the measurement with respect to sensor bias states.

        The star tracker bias is modeled as additive but is **not included**
        in the estimator state for this sensor.

        Returns
        -------
        ndarray, shape (0, 3)
            Empty bias Jacobian.
        """
        return np.zeros((0, self.output_length))

    @property
    def noise_covariance(self) -> NDArray[np.float64]:
        r"""
        Measurement noise covariance matrix.

        Returns
        -------
        ndarray, shape (3, 3)
            Noise covariance expressed in the spacecraft body frame.
        """
        if self.noise:
            return self.noise.cov()
        return np.zeros((3, 3))