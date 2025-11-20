__all__ = ["Estimator"]

import numpy as np
from typing import List, Optional
from ADCS.orbits.orbital_state import Orbital_State, Ephemeris
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.helpers.math_helpers import square_mat_sections, wahbas_svd

class Estimator():
    r"""
    Abstract base class for spacecraft attitude and bias estimators.

    This class encapsulates common bookkeeping for estimators operating on an
    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    model, including:

    - holding the **full augmented state vector**,
    - maintaining its **covariance** and **process noise** in a reduced
      representation (3-DOF attitude),
    - interfacing with the underlying satellite model via
      :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

    The augmented state has the structure

    .. math::

        \mathbf{x} =
        \begin{bmatrix}
            \boldsymbol{\omega} \\
            \mathbf{q} \\
            \mathbf{h}_{RW} \\
            \mathbf{b}_{act} \\
            \mathbf{b}_{sens} \\
            \boldsymbol{\theta}_{dist}
        \end{bmatrix}
        \in \mathbb{R}^{N},

    where:

    - :math:`\boldsymbol{\omega} \in \mathbb{R}^3` is the body angular rate,
    - :math:`\mathbf{q} \in \mathbb{R}^4` is the attitude quaternion,
    - :math:`\mathbf{h}_{RW} \in \mathbb{R}^{n_{RW}}` are reaction–wheel momenta,
    - :math:`\mathbf{b}_{act} \in \mathbb{R}^{n_{ab}}` are actuator bias states,
    - :math:`\mathbf{b}_{sens} \in \mathbb{R}^{n_{sb}}` are attitude–sensor biases,
    - :math:`\boldsymbol{\theta}_{dist} \in \mathbb{R}^{n_{dp}}` are disturbance parameters.

    Thus the total dimension is

    .. math::

        N = 7 + n_{RW} + n_{ab} + n_{sb} + n_{dp}.

    The **covariance matrices** are stored in a *reduced space* of dimension
    :math:`N-1`, corresponding to a 3-parameter attitude representation
    (e.g., error-angle vector) instead of the 4-parameter quaternion.
    The mapping between full and reduced coordinates is handled by subclasses
    (e.g. MEKF, SRUKF).

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Estimated satellite model that defines the structure of the state
        (number of reaction wheels, bias lengths, disturbance parameters).
    J2000 : float
        Initial time (seconds since J2000) used to construct an initial
        :class:`~ADCS.orbits.orbital_state.Orbital_State`. Only stored here;
        specific estimators may use it differently.
    x_hat : numpy.ndarray, shape (N,)
        Initial augmented state estimate :math:`\mathbf{x}_0`.
        Must satisfy:

        .. math::

            \mathrm{len}(x\_hat)
            =
            7 + n_{RW} + n_{ab} + n_{sb} + n_{dp}.
    P_hat : numpy.ndarray, shape (N-1, N-1)
        Initial covariance matrix in the **reduced** state space
        (3-parameter attitude representation).
    Q_hat : numpy.ndarray, shape (N-1, N-1)
        Process noise covariance in the same reduced space as ``P_hat``.
    dt : float, optional
        Estimator propagation time step :math:`\Delta t` [s]. Default is 1.0.
    cross_term : bool, optional
        If ``False``, cross–covariance terms between certain bias/parameter
        blocks may be explicitly zeroed in :meth:`update` to impose a block
        structure. If ``True``, the full covariance is kept.

    Attributes
    ----------
    est_sat : EstimatedSatellite
        Underlying satellite model used to propagate dynamics and predict
        measurements.
    x_hat : EstimatedArray
        Current estimated state, covariance and process noise.
        See :class:`~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedArray`.
    x0_hat : EstimatedArray
        Initial estimate (copy of the input at reset time).
    use : numpy.ndarray of bool, shape (N,)
        Boolean mask indicating which elements of the full state are active
        in the estimation. Used by :meth:`cov_use`.
    prev_os : Orbital_State
        Last orbital state used for propagation; some estimators may interpolate
        between ``prev_os`` and the current :class:`Orbital_State`.
    len_before_sens_bias : int
        Convenience length

        .. math::

            \text{len\_before\_sens\_bias} =
            \text{state\_len} + n_{ab},

        i.e., number of entries in :math:`[\boldsymbol{\omega},\mathbf{q},\mathbf{h}_{RW},\mathbf{b}_{act}]`.
    dt : float
        Estimator time step.
    cross_term : bool
        Whether to keep or zero cross–covariance blocks between certain
        bias/parameter groups.
    """
    def __init__(self, est_sat: EstimatedSatellite, J2000: float, x_hat: np.ndarray, P_hat: np.ndarray, Q_hat: np.ndarray, dt: float = 1, cross_term: bool = False) -> None:
        self.est_sat = est_sat
        self.cross_term = cross_term
        self.dt = dt
        self.len_before_sens_bias = self.est_sat.state_len + self.est_sat.act_bias_len

        ephem = Ephemeris()
        self.prev_os = Orbital_State(ephem=ephem, J2000=0, R=np.array([1,0,0]), V=np.array([0,1,0]))

        self.reset(J2000=J2000, x_hat=x_hat, P_hat=P_hat, Q_hat=Q_hat, dt=dt, cross_term=cross_term)

    def reset(self, J2000: float, x_hat: np.ndarray, P_hat: np.ndarray, Q_hat: np.ndarray, dt: float = 1, cross_term: bool = False) -> None:
        r"""
        Reset the estimator state, covariance, and process noise.

        This method enforces dimensional consistency between the augmented
        state vector and the reduced covariance/process-noise matrices, then
        stores them into :class:`EstimatedArray` objects and synchronizes the
        internal :class:`EstimatedSatellite` model via
        :meth:`EstimatedSatellite.match_estimate`.

        Let

        .. math::

            N = 7 + n_{RW} + n_{ab} + n_{sb} + n_{dp},

        as defined in the class docstring.

        Parameters
        ----------
        J2000 : float
            Time in seconds since J2000 corresponding to the new reset.
            Currently stored but not directly used by this base class.
        x_hat : numpy.ndarray, shape (N,)
            New augmented state estimate :math:`\mathbf{x}`.
        P_hat : numpy.ndarray, shape (N-1, N-1)
            Covariance matrix :math:`P` in reduced coordinates,
            where the quaternion is represented by a 3-parameter attitude
            error vector.
        Q_hat : numpy.ndarray, shape (N-1, N-1)
            Process noise covariance :math:`Q` in the same reduced space
            as ``P_hat``.
        dt : float, optional
            Time step :math:`\Delta t` [s] for propagation. Default is 1.0.
        cross_term : bool, optional
            Flag controlling whether cross–covariance terms between actuator
            bias, sensor bias, and disturbance parameter blocks should be
            preserved (``True``) or zeroed (``False``) in :meth:`update`.

        Raises
        ------
        ValueError
            If ``x_hat`` has inconsistent length, or if ``P_hat``/``Q_hat``
            do not have shape ``(len(x_hat) - 1, len(x_hat) - 1)``.

        Notes
        -----
        After validation, the method:

        1. Creates :class:`EstimatedArray` ``x0_hat`` and ``x_hat`` objects.
        2. Sets the mask :attr:`use` to all ``True`` (all state entries active).
        3. Calls :meth:`EstimatedSatellite.match_estimate` to update the
           internal satellite model with the new estimate.
        """
        # Verify x_hat vector
        if len(x_hat) != 7 + self.est_sat.number_RW + self.est_sat.act_bias_len + self.est_sat.att_sens_bias_len + self.est_sat.dist_param_len:
            raise ValueError("x_hat length does not match estimate length in EstimatedSatellite")
        
        # Verify P_hat matrix. It should be one shorter than x_hat, since x_hat uses quaternions, which are 3 DOF
        if P_hat.shape != (len(x_hat) - 1, len(x_hat) - 1):
            raise ValueError("P_hat shape does not match MEKF x_hat")
        
        # Verify Q_hat matrix. It should be one shorter than x_hat
        if Q_hat.shape != (len(x_hat) - 1, len(x_hat) - 1):
            raise ValueError("Q_hat shape does not match MEKF x_hat")
        
        self.x0_hat = EstimatedArray(val=x_hat, cov=P_hat, int_cov=Q_hat)
        # Full estimated state
        self.x_hat = self.x0_hat.copy()

        self.use = np.ones(self.x_hat.val.size).astype(bool)
        self.state_len = len(self.use)
        # Update estimated satellite
        self.est_sat.match_estimate(est_state=self.x_hat, dt=dt)


    def initialize_estimate(
        self,
        gyro_readings: Optional[np.ndarray],
        mtm_readings: Optional[np.ndarray],
        sunpair_readings: Optional[np.ndarray],
        sunsensor_readings: Optional[np.ndarray],
        B_ECI: Optional[np.ndarray],
        S_ECI: Optional[np.ndarray],
        os: Orbital_State,
        q: Optional[np.ndarray] = None
    ):
        r"""
        Robust initial attitude and bias estimation.

        This routine builds an initial attitude and gyro–bias estimate using
        the currently stored covariance and a subset of available sensor
        measurements. It is designed to work even if some sensors are missing
        (passed as ``None``).

        The attitude is estimated by solving **Wahba's problem** using all
        available vector observations (magnetometer, sun sensor, sun sensor
        pair). Each vector is weighted by an inverse-variance-like term that
        combines sensor noise variance and the current prior bias uncertainty.

        If gyroscope data is provided, a simple linear estimator is used to
        obtain an initial gyro bias and angular rate.

        Parameters
        ----------
        gyro_readings : numpy.ndarray or None, shape (3,), optional
            Gyroscope measurement :math:`\tilde{\boldsymbol{\omega}}_b` in the
            body frame. If ``None``, gyro bias and rate are not updated.
        mtm_readings : numpy.ndarray or None, shape (3,), optional
            Magnetometer measurement in body frame. Used together with
            :math:`\mathbf{B}_{ECI}` to form one Wahba vector pair.
        sunpair_readings : numpy.ndarray or None, shape (3,), optional
            Combined sun sensor pair reading in body frame, if available.
        sunsensor_readings : numpy.ndarray or None, shape (3,), optional
            Simple sun sensor reading in body frame.
        B_ECI : numpy.ndarray or None, shape (3,), optional
            Reference magnetic field vector in ECI/frame used for Wahba.
        S_ECI : numpy.ndarray or None, shape (3,), optional
            Reference sun direction vector in ECI/frame used for Wahba.
        os : Orbital_State
            Current orbital state (not directly used here, but included for
            interface consistency and potential future extensions).
        q : numpy.ndarray or None, shape (4,), optional
            If provided, this quaternion is used as the initial attitude.
            If ``None``, Wahba's problem is solved via
            :func:`~ADCS.helpers.math_helpers.wahbas_svd` to obtain :math:`q`.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Updated augmented state vector with the new attitude and
            optionally gyro bias and angular rate.

        Raises
        ------
        RuntimeError
            If no valid vector measurements (magnetometer or any sun sensor)
            are available to form at least one Wahba vector pair.

        Notes
        -----
        **Wahba step**

        For each available vector measurement :math:`\mathbf{v}_b` and its
        reference counterpart :math:`\mathbf{v}_{ECI}`, a weight is computed as

        .. math::

            w_i = \frac{1}{\sigma_i^2 + P_{\text{bias},i}},

        where :math:`\sigma_i^2` is the sensor noise variance (scaled) and
        :math:`P_{\text{bias},i}` is the prior bias variance extracted from
        the current covariance matrix.

        These triples ``(w_i, v_{b,i}, v_{ECI,i})`` are passed to
        :func:`~ADCS.helpers.math_helpers.wahbas_svd` to compute the
        attitude quaternion.

        **Gyro bias**

        If ``gyro_readings`` is not ``None``, the 3×3 covariance block for the
        gyro bias, the measurement noise covariance, and the current attitude–
        rate covariance are combined to form a simple linear estimator for the
        bias and an approximate initial angular rate. The corresponding entries
        in :attr:`x_hat.val` are updated and pushed to the satellite model via
        :meth:`EstimatedSatellite.match_estimate`.
        """

        # ---- 1. Build lists of available vectors for Wahba ----------------------

        body_vecs = []
        eci_vecs  = []
        weights   = []

        # ---- Magnetometer ----
        if mtm_readings is not None and B_ECI is not None:
            sens = self.est_sat.mtm_sensor   # You will have this reference
            var = sens.noise.std_noise**2 * sens.scale**2

            bias_idx = sens.bias_state_index  # precomputed during satellite setup
            P_bias = self.x_hat.cov[bias_idx, bias_idx]

            body_vecs.append(mtm_readings)
            eci_vecs.append(B_ECI)
            weights.append(1.0 / (var + P_bias))

        # ---- Sun sensor (simple) ----
        if sunsensor_readings is not None and S_ECI is not None:
            sens = self.est_sat.sun_sensor
            var  = sens.noise.std_noise**2 * sens.scale**2
            bias_idx = sens.bias_state_index
            P_bias = self.x_hat.cov[bias_idx, bias_idx]

            body_vecs.append(sunsensor_readings)
            eci_vecs.append(S_ECI)
            weights.append(1.0 / (var + P_bias))

        # ---- Sun sensor pair ----
        if sunpair_readings is not None and S_ECI is not None:
            sens = self.est_sat.sunpair_sensor
            var  = sens.noise.std_noise**2 * sens.scale**2
            bias_idx = sens.bias_state_index
            P_bias = self.x_hat.cov[bias_idx, bias_idx]

            body_vecs.append(sunpair_readings)
            eci_vecs.append(S_ECI)
            weights.append(1.0 / (var + P_bias))

        # Must have >= 1 vector to solve Wahba
        if len(body_vecs) == 0:
            raise RuntimeError("No valid vector measurements available at initialization")

        # ---- 2. Solve Wahba (if q not provided) ---------------------------------

        if q is None:
            q = wahbas_svd(weights, body_vecs, eci_vecs)

        # ----------------------------------------------------------
        # Insert quaternion into full state
        # ----------------------------------------------------------
        state = self.x_hat.val.copy()
        state[3:7] = q

        # ----------------------------------------------------------
        # 3. Estimate biases (robust to missing sensors)
        # ----------------------------------------------------------

        # Gyro bias
        if gyro_readings is not None:
            sens = self.est_sat.gyro_sensor
            idx0 = sens.bias_state_index
            idx1 = idx0 + 3

            cov_g = self.x_hat.cov[idx0:idx1, idx0:idx1]
            R_g   = np.eye(3) * sens.noise.std_noise**2
            Qw    = self.x_hat.cov[0:3, 0:3]  # rotational rate covariance

            b_gyro = np.linalg.inv(R_g + cov_g + Qw) @ cov_g @ gyro_readings
            state[idx0:idx1] = b_gyro

            # Estimate angular rate
            wguess = Qw @ np.linalg.inv(cov_g) @ b_gyro
            state[0:3] = wguess

        # ----------------------------------------------------------
        # Commit state back
        # ----------------------------------------------------------
        self.x_hat.val = state
        self.est_sat.match_estimate(self.x_hat, dt=1.0)

        return state


    def update(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> np.ndarray:
        r"""
        High-level estimator update wrapper.

        This method delegates the actual filtering step to a subclass‐defined
        :meth:`update_core` method (e.g., EKF, SRUKF), then optionally enforces
        a block-structured covariance by zeroing cross terms between selected
        bias/parameter groups, updates the internal :class:`EstimatedArray`
        instance, and synchronizes the associated
        :class:`EstimatedSatellite` model.

        Parameters
        ----------
        u : numpy.ndarray, shape (n_u,)
            Control input vector applied to the satellite (reaction wheel
            commands, thruster commands, etc.).
        sensors : list of numpy.ndarray
            Collection of sensor measurements used by the estimator. The
            contents, order, and stacking convention are determined by the
            subclass implementation of :meth:`update_core` and the satellite
            measurement model (e.g. :meth:`EstimatedSatellite.noiseless_sensor_readings`).
        os : Orbital_State
            Current orbital/environmental state used for propagation and
            measurement prediction.

        Returns
        -------
        numpy.ndarray, shape (N_phys,)
            Updated estimate of the **physical satellite state**, typically
            the first :attr:`est_sat.state_len` entries of the augmented state
            (i.e., :math:`[\boldsymbol{\omega}, \mathbf{q}, \mathbf{h}_{RW}]`),
            as returned by :meth:`update_core`. The full augmented estimate is
            stored in :attr:`x_hat`.

        Notes
        -----
        1. The subclass :meth:`update_core` is expected to return a new
           :class:`EstimatedArray` ``x_hat`` with:

           - ``x_hat.val`` : full augmented state,
           - ``x_hat.cov`` : reduced covariance in :math:`\mathbb{R}^{N-1}`,
           - ``x_hat.int_cov`` : process-noise covariance (typically unchanged).

        2. If :attr:`prev_os` has not yet been initialized (``J2000 == 0``),
           it is set to ``os`` in this call.

        3. If :attr:`cross_term` is ``False``, cross-covariance blocks between
           actuator biases, sensor biases and disturbance parameters are set to
           zero to enforce a block-diagonal structure. The exact indices are
           derived from :attr:`est_sat.state_len`, :attr:`est_sat.act_bias_len`,
           :attr:`est_sat.att_sens_bias_len`, and :attr:`est_sat.dist_param_len`.

        4. The final estimate is inserted back into the main
           :attr:`x_hat` via :meth:`EstimatedArray.set_indices` using a
           reduced covariance mask given by :meth:`cov_use`. Afterwards
           :meth:`EstimatedSatellite.match_estimate` is called to update the
           satellite model (e.g., biases, disturbance parameters).
        """
        if self.prev_os.J2000 == 0:
            self.prev_os = os

        x_hat: EstimatedArray = self.update_core(u=u, sensors=sensors, os=os)

        oc = x_hat.cov
        if not self.cross_term:
            ab0 = self.est_sat.state_len - 1
            ab1 = ab0 + self.est_sat.act_bias_len

            sb0 = ab1
            sb1 = sb0 + self.est_sat.att_sens_bias_len
            d0 = sb1

            oc[ab0:ab1,sb0:sb1] = 0
            oc[sb0:sb1,ab0:ab1] = 0
            oc[ab0:ab1,d0:] = 0
            oc[d0:,ab0:ab1] = 0
            oc[sb0:sb1,d0:] = 0
            oc[d0:,sb0:sb1] = 0

        self.x_hat.set_indices(self.use, x_hat.val, oc, square_mat_sections(self.x_hat.int_cov, self.cov_use()),[3])
        self.est_sat.match_estimate(self.x_hat, self.dt)

        return self.x_hat.val


    def cov_use(self) -> np.ndarray:
        r"""
        Return a boolean mask for the reduced covariance state.

        The :attr:`use` array is a boolean mask over the **full** augmented
        state :math:`\mathbf{x} \in \mathbb{R}^{N}`. The covariance is stored
        in a reduced state space of dimension :math:`N-1`, where the quaternion
        has been replaced by a 3-parameter representation.

        This method removes the entry corresponding to the dropped quaternion
        component (index 3 in the full state), yielding a mask compatible with
        the reduced covariance.

        Returns
        -------
        numpy.ndarray of bool, shape (N-1,)
            Boolean mask indicating which elements of the reduced state are
            active in the covariance. The length matches the dimension of
            :attr:`x_hat.cov` and :attr:`x_hat.int_cov`.

        Notes
        -----
        If the full state is indexed as:

        .. math::

            \mathbf{x} =
            \begin{bmatrix}
                \omega_x \\ \omega_y \\ \omega_z \\
                q_0 \\ q_1 \\ q_2 \\ q_3 \\
                \vdots
            \end{bmatrix},

        then index 3 corresponds to the scalar quaternion component that is
        dropped in the reduced state. This function simply deletes that index
        from :attr:`use` to obtain a mask for the reduced covariance.
        """
        res = self.use.copy()
        res = np.delete(res, 3)
        return res