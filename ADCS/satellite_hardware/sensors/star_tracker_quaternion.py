__all__ = ["StarTrackerQuaternion"]

import numpy as np
from typing import Optional, List
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import rot_mat

from ADCS.orbits.orbital_state import Orbital_State


class StarTrackerQuaternion(Sensor):
    r"""
    Star tracker sensor with quaternion attitude output.

    Unlike the single-vector :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`,
    this sensor solves for the full spacecraft attitude by observing **multiple
    navigation stars** simultaneously and applying Wahba's attitude determination
    algorithm.

    Measurement model
    -----------------
    Given :math:`N \ge 2` visible stars with known inertial directions
    :math:`\{\mathbf{r}_i\}` and measured body-frame directions
    :math:`\{\mathbf{b}_i\}`, the sensor solves Wahba's problem:

    .. math::

        \hat{\mathbf{C}} = \arg\min_{\mathbf{C}}
        \frac{1}{2} \sum_{i=1}^{N} w_i
        \left\| \mathbf{b}_i - \mathbf{C}\,\mathbf{r}_i \right\|^2

    The optimal rotation matrix :math:`\hat{\mathbf{C}}` is obtained via SVD
    of the attitude profile matrix and converted to a quaternion:

    .. math::

        \mathbf{q}_{\text{meas}} = \mathrm{quat}(\hat{\mathbf{C}})

    Star weighting uses inverse visual magnitude so that brighter stars
    receive higher weight.

    Including bias and noise, the full measurement is

    .. math::

        \tilde{\mathbf{q}} = \mathbf{q}_{\text{clean}} + \mathbf{b} + \mathbf{n}

    After corruption, the quaternion is renormalized and the scalar component
    is enforced positive.

    Star selection
    --------------
    At each measurement epoch, the sensor:

    1. Projects the sensor boresight into the inertial frame
    2. Queries the :class:`~ADCS.environment.StarCatalog` for visible stars
    3. Applies field-of-view and exclusion constraints (Sun, optional Moon)
    4. Requires at least ``min_stars`` visible stars for a valid solution

    If fewer than ``min_stars`` stars are available, the output is ``NaN``.

    Estimator properties
    --------------------
    * Output dimension: 4 (unit quaternion)
    * Depends only on attitude quaternion
    * Jacobian is nonzero only w.r.t. quaternion states
    * No coupling to angular velocity or momentum states

    See Also
    --------
    :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`
    :class:`~ADCS.environment.StarCatalog`
    :func:`~ADCS.helpers.math_helpers.wahbas_svd`
    """

    output_length: int = 4

    def __init__(
        self,
        sample_time: float = 0.1,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
        boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
        fov: float = np.deg2rad(20.0),
        sun_exclusion: float = np.deg2rad(25.0),
        min_stars: int = 2,
        star_catalog: Optional[StarCatalog] = None,
    ) -> None:
        r"""
        Initialize the quaternion star tracker sensor.

        :param sample_time: Sampling period [s].
        :type sample_time: float
        :param bias: Optional additive bias model (4-element).
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None
        :param noise: Optional noise model (4-element).
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None
        :param estimate_bias: If ``True``, bias is included in the estimator state.
        :type estimate_bias: bool
        :param boresight: Sensor boresight direction in the body frame, shape ``(3,)``.
                          Normalized internally.
        :type boresight: numpy.ndarray
        :param fov: Full-angle field of view [rad].  Should be wide enough to
                    observe at least ``min_stars`` stars simultaneously.
        :type fov: float
        :param sun_exclusion: Minimum allowable Sun-boresight separation [rad].
        :type sun_exclusion: float
        :param min_stars: Minimum number of visible stars required for a valid
                          attitude solution.
        :type min_stars: int
        :param star_catalog: Navigation star catalog.  If ``None``, the default
                             :class:`~ADCS.environment.StarCatalog` is used.
        :type star_catalog: :class:`~ADCS.environment.StarCatalog` or None
        :return: None
        :rtype: None
        """
        self.boresight = np.asarray(boresight, dtype=np.float64)
        norm = np.linalg.norm(self.boresight)
        if norm < 1e-6:
            raise ValueError("Boresight vector cannot be zero.")
        self.boresight = self.boresight / norm

        self.fov = float(fov)
        self.sun_exclusion = float(sun_exclusion)
        self.min_stars = int(min_stars)

        self.catalog = star_catalog if star_catalog is not None else StarCatalog()
        self.current_stars: List[NavigationStar] = []

        super().__init__(
            sample_time=sample_time,
            output_length=4,
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias,
        )

    # ------------------------------------------------------------------
    # Internal helpers (reuse StarTracker patterns)
    # ------------------------------------------------------------------

    def _get_sun_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        if hasattr(os, "S") and os.S is not None:
            s = np.asarray(os.S, dtype=np.float64)
            if not np.allclose(s, 0):
                return s
        return None

    def _get_moon_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        try:
            if hasattr(os, "ephem") and os.ephem is not None:
                moon = os.ephem.planets["moon"]
                moon_icrf = os.ephem.earth.at(os.sf_pos.t).observe(moon).apparent()
                return np.asarray(moon_icrf.position.km, dtype=np.float64)
        except (KeyError, AttributeError):
            pass
        return None

    def _select_stars(
        self, q: NDArray[np.float64], os: Orbital_State
    ) -> List[NavigationStar]:
        r"""
        Select all visible navigation stars sorted by brightness.

        :param q: Attitude quaternion (body -> inertial), shape ``(4,)``.
        :type q: numpy.ndarray
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Visible navigation stars sorted by ascending visual magnitude.
        :rtype: list[:class:`~ADCS.environment.NavigationStar`]
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
            sun_exclusion_rad=self.sun_exclusion,
        )

        return sorted(visible, key=lambda s: s.vmag)

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def clean_reading(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Compute the noise-free quaternion attitude measurement.

        The method determines star visibility and, if at least ``min_stars``
        navigation stars are observable, returns the true attitude quaternion.
        In the noise-free case, the star tracker perfectly recovers the
        spacecraft attitude from the observed star directions.

        The visibility check is the key physical constraint: the sensor
        can only provide a measurement when sufficient stars fall within
        its field of view and are not occluded by the Earth, Moon, or Sun.

        .. math::

            \mathbf{q}_{\text{clean}} = \mathbf{q}

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state used for star visibility.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Attitude quaternion (scalar-first), or ``NaN`` if insufficient
                 stars are visible.
        :rtype: numpy.ndarray
        """
        q = x.q.copy()
        stars = self._select_stars(q, os)

        if len(stars) < self.min_stars:
            self.current_stars = []
            return np.full(4, np.nan)

        self.current_stars = stars

        # Enforce scalar-positive convention
        if q[0] < 0:
            q = -q

        return q

    def reading(
        self,
        x: NDArray[np.float64],
        os: Orbital_State,
        dmode: Optional[ErrorMode] = None,
    ) -> NDArray[np.float64]:
        r"""
        Compute the full quaternion measurement including bias and noise.

        After the base class applies bias and noise, the quaternion is
        renormalized and the scalar component is enforced positive.

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param dmode: Error mode controlling bias and noise application.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` or None
        :return: Normalized attitude quaternion measurement.
        :rtype: numpy.ndarray
        """
        measurement = super().reading(x, os, dmode)

        if not np.any(np.isnan(measurement)):
            norm = np.linalg.norm(measurement)
            if norm > 1e-9:
                measurement = measurement / norm
            if measurement[0] < 0:
                measurement = -measurement

        return measurement

    def basestate_jac(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Jacobian of the quaternion measurement w.r.t. the base state.

        The measurement depends only on the attitude quaternion. The angular
        velocity block is zero.

        For the clean measurement (Wahba's solution from noise-free star
        observations), the measured quaternion equals the true quaternion.
        Therefore the Jacobian w.r.t. the quaternion states is the identity:

        .. math::

            \frac{\partial \mathbf{q}_{\text{meas}}}{\partial \boldsymbol{\omega}}
            = \mathbf{0}_{3 \times 4},
            \qquad
            \frac{\partial \mathbf{q}_{\text{meas}}}{\partial \mathbf{q}}
            = \mathbf{I}_{4 \times 4}

        :param x: Full spacecraft state vector.
        :type x: ADCS.state.State
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Base-state Jacobian of shape ``(7, 4)``.
        :rtype: numpy.ndarray
        """
        if not self.current_stars:
            return np.zeros((7, self.output_length))

        J = np.zeros((7, 4))
        J[3:7, :] = np.eye(4)
        return J

    def bias_jac(
        self, x: NDArray[np.float64], os: Orbital_State
    ) -> NDArray[np.float64]:
        r"""
        Jacobian of the measurement w.r.t. bias states.

        The quaternion star tracker bias is not included in the estimator
        state, so the bias Jacobian is empty.

        :param x: Full spacecraft state vector (unused).
        :type x: ADCS.state.State
        :param os: Orbital state (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Empty bias Jacobian of shape ``(0, 4)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((0, self.output_length))
