__all__ = ["StarTrackerQuaternion"]

import numpy as np
from typing import Optional, List
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import rot_mat, quat_mult

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
        earth_limb_exclusion: float = 0.0,
        max_rate: Optional[float] = None,
        sigma_cross: Optional[float] = None,
        sigma_roll: Optional[float] = None,
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
        :param earth_limb_exclusion: Boresight keep-out **beyond** the geometric Earth
            limb [rad]. The catalog already drops individual stars behind the Earth; this
            is the separate, and usually binding, constraint that the tracker refuses to
            produce a solution when its boresight is within this margin of the limb
            (stray light). ``0.0`` reproduces the previous behaviour.
        :type earth_limb_exclusion: float
        :param max_rate: Body-rate magnitude above which the tracker drops out [rad/s]
            (image smear). ``None`` disables the check and reproduces the previous
            behaviour.
        :type max_rate: float or None
        :param sigma_cross: 1-sigma attitude error about the two axes perpendicular to the
            boresight [rad]. When given (with ``sigma_roll``), the measurement is perturbed
            by a small rotation rather than by additive quaternion noise, so the
            cross-boresight/roll anisotropy real trackers quote is represented.
        :type sigma_cross: float or None
        :param sigma_roll: 1-sigma attitude error **about** the boresight [rad]. Typically
            several times ``sigma_cross``.
        :type sigma_roll: float or None
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
        self.earth_limb_exclusion = float(earth_limb_exclusion)
        self.max_rate = None if max_rate is None else float(max_rate)
        self.sigma_cross = None if sigma_cross is None else float(sigma_cross)
        self.sigma_roll = None if sigma_roll is None else float(sigma_roll)
        if (self.sigma_cross is None) != (self.sigma_roll is None):
            raise ValueError(
                "sigma_cross and sigma_roll must be given together (anisotropic mode) "
                "or both omitted (additive-quaternion-noise mode)."
            )
        #: True when the anisotropic small-rotation error model is active.
        self.anisotropic: bool = self.sigma_cross is not None

        #: Whether the most recent :meth:`clean_reading` produced a solution. Campaigns
        #: log the per-trial time-average of this as "tracker-available fraction".
        self.available: bool = False

        # Body-frame triad with the boresight as the third axis, so the anisotropic
        # perturbation can be drawn in sensor axes and rotated into the body frame.
        seed = (np.array([1.0, 0.0, 0.0])
                if abs(self.boresight[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        e1 = np.cross(seed, self.boresight)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.boresight, e1)
        self._sensor_axes = np.column_stack((e1, e2, self.boresight))

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
        :type x: numpy.ndarray
        :param os: Orbital state used for star visibility.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Attitude quaternion (scalar-first), or ``NaN`` if insufficient
                 stars are visible.
        :rtype: numpy.ndarray
        """
        q = x[3:7].copy()

        # Slew-rate dropout: above max_rate the star images smear and the tracker
        # produces no solution. This is the constraint that couples tracker
        # availability to agility, so campaigns that sweep slew rate must model it.
        if self.max_rate is not None and np.size(x) >= 3:
            if float(np.linalg.norm(np.ravel(x)[0:3])) > self.max_rate:
                self.current_stars = []
                self.available = False
                return np.full(4, np.nan)

        # Boresight Earth-limb keep-out (stray light). The catalog drops individual
        # stars behind the geometric limb; this is the separate boresight margin.
        if self.earth_limb_exclusion > 0.0:
            R = np.asarray(getattr(os, "R", None), dtype=np.float64)
            if R is not None and R.size == 3:
                r = float(np.linalg.norm(R))
                if r > self.catalog.R_EARTH:
                    nadir = -R / r
                    b_eci = rot_mat(q) @ self.boresight
                    limb = np.arcsin(np.clip(self.catalog.R_EARTH / r, -1.0, 1.0))
                    from_nadir = np.arccos(
                        np.clip(float(np.dot(b_eci, nadir)), -1.0, 1.0)
                    )
                    if from_nadir < limb + self.earth_limb_exclusion:
                        self.current_stars = []
                        self.available = False
                        return np.full(4, np.nan)

        stars = self._select_stars(q, os)

        if len(stars) < self.min_stars:
            self.current_stars = []
            self.available = False
            return np.full(4, np.nan)

        self.current_stars = stars
        self.available = True

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
        :type x: numpy.ndarray
        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param dmode: Error mode controlling bias and noise application.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.ErrorMode` or None
        :return: Normalized attitude quaternion measurement.
        :rtype: numpy.ndarray
        """
        if self.anisotropic:
            # Anisotropic mode: perturb by a small rotation in sensor axes rather than
            # adding noise to the quaternion components. A tracker's error is quoted as
            # cross-boresight vs about-boresight and the two differ by ~6x; additive
            # quaternion noise cannot represent that. The roll term lands on the
            # boresight axis.
            #
            # This mirrors the base Sensor.reading() pipeline rather than calling it,
            # because only the *noise* step changes. dmode must still be honoured:
            # estimators request clean predictions with add_noise=False, and adding
            # noise there would corrupt sigma-point propagation.
            if dmode is None:
                dmode = ErrorMode(add_bias=True, add_noise=True,
                                  update_bias=True, update_noise=True)

            measurement = self.clean_reading(x=x, os=os)

            if dmode.update_bias:
                self.bias._update_bias(os.J2000)
            if self.bias and dmode.add_bias:
                measurement = measurement + self.bias.get_bias(os.J2000)

            # Keep the noise process ticking even when the sample is not applied, so
            # a run's stochastic stream does not depend on how many clean predictions
            # the estimator happened to ask for.
            if dmode.update_noise:
                self.noise._update_noise()
            if dmode.add_noise and not np.any(np.isnan(measurement)):
                d = np.array([
                    self.sigma_cross * np.random.randn(),
                    self.sigma_cross * np.random.randn(),
                    self.sigma_roll * np.random.randn(),
                ])
                dtheta = self._sensor_axes @ d          # sensor -> body
                dq = np.concatenate(([1.0], 0.5 * dtheta))
                dq = dq / np.linalg.norm(dq)
                measurement = quat_mult(measurement, dq)
        else:
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
        :type x: numpy.ndarray
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
        :type x: numpy.ndarray
        :param os: Orbital state (unused).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Empty bias Jacobian of shape ``(0, 4)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((0, self.output_length))
