__all__ = ["Attitude_Estimator"]

import numpy as np
import time
from typing import List, Optional, Sequence
from ADCS.orbits.orbital_state import Orbital_State, Ephemeris
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.state import EstimatorState
from ADCS.helpers.math_helpers import square_mat_sections, wahbas_svd

class Attitude_Estimator():
    r"""
    Abstract base class for spacecraft attitude and bias estimators.

    This base class provides shared bookkeeping and interfaces for estimators
    operating on an :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    model, including:

    - storage of the **full augmented state vector**,
    - storage of covariance and process-noise matrices in a **reduced attitude space**
      (3-DOF attitude error instead of a 4-parameter quaternion),
    - synchronization of the internal satellite model via
      :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

    Augmented state definition
    --------------------------
    The estimator maintains an augmented state of the form

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

    .. math::

        \boldsymbol{\omega} \in \mathbb{R}^3,\quad
        \mathbf{q} \in \mathbb{R}^4,\quad
        \mathbf{h}_{RW} \in \mathbb{R}^{n_{RW}},\quad
        \mathbf{b}_{act} \in \mathbb{R}^{n_{ab}},\quad
        \mathbf{b}_{sens} \in \mathbb{R}^{n_{sb}},\quad
        \boldsymbol{\theta}_{dist} \in \mathbb{R}^{n_{dp}}.

    Hence the total augmented dimension is

    .. math::

        N = 7 + n_{RW} + n_{ab} + n_{sb} + n_{dp}.

    Reduced covariance representation
    ---------------------------------
    The state contains a quaternion :math:`\mathbf{q}\in\mathbb{R}^4`, but the
    covariance is typically stored in a **minimal 3-DOF attitude-error space**
    to avoid the unit-norm constraint in the covariance representation. Thus,
    the covariance and process noise live in

    .. math::

        \mathbb{R}^{(N-1)\times(N-1)}

    where one quaternion component is dropped and replaced by a 3-parameter
    attitude error (e.g., error-angle vector). Subclasses (e.g., MEKF, SRUKF)
    define the mapping between the full state and this reduced covariance space.

    If ``quat_as_vec=True``, then the covariance and process noise are stored
    in the **full** space (no reduction), i.e. shape :math:`N\times N`.

    References
    ----------
    - Satellite model synchronization is performed with
      :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.
    - Reduced covariance sub-selection is performed with
      :func:`~ADCS.helpers.math_helpers.square_mat_sections`.

    """
    def __init__(self, est_sat: EstimatedSatellite, J2000: float, x_hat: EstimatorState, P_hat: np.ndarray, Q_hat: np.ndarray, dt: float = 1, cross_term: bool = False, quat_as_vec: bool = False, ephem: Optional[Ephemeris] = None) -> None:
        r"""
        Construct an attitude estimator and initialize its internal bookkeeping.

        This initializer stores the satellite model reference, prepares a
        placeholder previous orbital state, and delegates all estimate/covariance
        initialization and validation to :meth:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator.reset`.

        Internally, a previous orbital state is created using
        :class:`~ADCS.orbits.orbital_state.Ephemeris` and
        :class:`~ADCS.orbits.orbital_state.Orbital_State` so that subsequent
        updates can safely rely on :attr:`prev_os`.

        :param est_sat: Estimated satellite model defining state structure
            (reaction wheels, bias lengths, disturbance parameters).
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param J2000: Initial time in seconds since J2000.
        :type J2000: float
        :param x_hat: Initial augmented state estimate :math:`\mathbf{x}_0` of length
            :math:`N = 7 + n_{RW} + n_{ab} + n_{sb} + n_{dp}`.
        :type x_hat: ADCS.state.EstimatorState
        :param P_hat: Initial covariance matrix, reduced to :math:`(N-1)\times(N-1)`
            unless ``quat_as_vec=True`` (then :math:`N\times N`).
        :type P_hat: numpy.ndarray
        :param Q_hat: Initial process-noise covariance matrix, same shape convention as ``P_hat``.
        :type Q_hat: numpy.ndarray
        :param dt: Estimator propagation time step :math:`\Delta t` in seconds.
        :type dt: float
        :param cross_term: If ``False``, certain cross-covariance blocks are zeroed
            in :meth:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator.update`.
        :type cross_term: bool
        :param quat_as_vec: If ``True``, store covariance and process noise in the full
            quaternion space (no reduction).
        :type quat_as_vec: bool
        :param ephem: Optional planetary ephemeris. If ``None``, one is constructed,
            which may download a 16 MB kernel on first use. Pass an existing
            instance to share it between objects or to avoid the download.
        :type ephem: ~ADCS.orbits.ephemeris.Ephemeris or None
        :return: ``None``.
        :rtype: None

        """
        self.est_sat = est_sat
        self.cross_term = cross_term
        self.quat_as_vec = quat_as_vec
        self.dt = dt
        self.len_before_sens_bias = self.est_sat.state_len + self.est_sat.act_bias_len

        # Constructing an Ephemeris can download a 16 MB kernel, so allow one
        # to be injected -- matching Orbital_State's `ephem=None` convention.
        # Without this the caller had no way to share a single ephemeris, nor
        # to point at one on a machine that cannot reach the network.
        ephem = Ephemeris() if ephem is None else ephem
        self.prev_os = Orbital_State(ephem=ephem, J2000=0, R=np.array([1,0,0]), V=np.array([0,1,0]))

        self.reset(J2000=J2000, x_hat=x_hat, P_hat=P_hat, Q_hat=Q_hat, dt=dt, cross_term=cross_term)

    def reset(self, J2000: float, x_hat: EstimatorState, P_hat: np.ndarray, Q_hat: np.ndarray, dt: float = 1, cross_term: bool = False) -> None:
        r"""
        Reset the estimator state, covariance, and process noise.

        This method validates dimensional consistency between the full augmented
        state vector :math:`\mathbf{x}\in\mathbb{R}^N` and the covariance/process-noise
        matrices, stores the result into :class:`~ADCS.state.EstimatorState`
        containers, and synchronizes the internal satellite model via
        :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

        Dimension consistency
        ---------------------
        The expected state length is

        .. math::

            N = 7 + n_{RW} + n_{ab} + n_{sb} + n_{dp},

        where the counts are taken from the supplied
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.

        Covariance representation follows one of two conventions:

        1. Reduced attitude space (default, ``quat_as_vec=False``):

           .. math::

               P,\;Q \in \mathbb{R}^{(N-1)\times(N-1)}.

        2. Full quaternion space (``quat_as_vec=True``):

           .. math::

               P,\;Q \in \mathbb{R}^{N\times N}.

        After validation, the reset procedure:

        - creates :class:`~ADCS.state.EstimatorState` instances
          for :attr:`x0_hat` and :attr:`x_hat`,
        - sets :attr:`use` to all ``True`` (all states active),
        - calls :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`
          to push the reset estimate into the satellite model.

        :param J2000: Time in seconds since J2000 corresponding to the new reset.
        :type J2000: float
        :param x_hat: New augmented state estimate :math:`\mathbf{x}` of length :math:`N`.
        :type x_hat: ADCS.state.EstimatorState
        :param P_hat: Covariance matrix, reduced or full depending on ``quat_as_vec``.
        :type P_hat: numpy.ndarray
        :param Q_hat: Process-noise covariance matrix, same shape convention as ``P_hat``.
        :type Q_hat: numpy.ndarray
        :param dt: Propagation time step :math:`\Delta t` in seconds used for satellite model synchronization.
        :type dt: float
        :param cross_term: Whether the estimator should preserve cross-covariance blocks
            between bias/parameter groups in :meth:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator.update`.
        :type cross_term: bool
        :raises ValueError: If ``x_hat`` length is inconsistent with the satellite model.
        :raises ValueError: If ``P_hat`` or ``Q_hat`` shapes do not match the required convention.
        :return: ``None``.
        :rtype: None

        """
        if not isinstance(x_hat, EstimatorState):
            raise TypeError(f"x_hat must be an EstimatorState, got {type(x_hat).__name__}")
        expected_x_hat_length = 7 + self.est_sat.number_RW + self.est_sat.act_bias_len + self.est_sat.att_sens_bias_len + self.est_sat.dist_param_len
        if x_hat.augmented_size != expected_x_hat_length:
            raise ValueError("x_hat length does not match estimate length in EstimatedSatellite")
        
        # Verify P_hat matrix. It should be one shorter than x_hat, since x_hat uses quaternions, which are 3 DOF
        if self.quat_as_vec:
            if P_hat.shape != (x_hat.augmented_size, x_hat.augmented_size):
                raise ValueError("P_hat shape does not match MEKF x_hat")
            if Q_hat.shape != (x_hat.augmented_size, x_hat.augmented_size):
                raise ValueError("Q_hat shape does not match MEKF x_hat")
        else:
            if P_hat.shape != (x_hat.augmented_size - 1, x_hat.augmented_size - 1):
                raise ValueError("P_hat shape does not match MEKF x_hat")
            if Q_hat.shape != (x_hat.augmented_size - 1, x_hat.augmented_size - 1):
                raise ValueError("Q_hat shape does not match MEKF x_hat")
        
        self.x0_hat = EstimatorState.from_estimator_array(
            x_hat.as_estimator_array(),
            n_rw=self.est_sat.number_RW,
            n_act_bias=self.est_sat.act_bias_len,
            n_sens_bias=self.est_sat.att_sens_bias_len,
            n_dist_param=self.est_sat.dist_param_len,
            cov=P_hat,
            int_cov=Q_hat,
        )
        # Full estimated state
        self.x_hat = self.x0_hat.copy()

        self.use = np.ones(self.x_hat.augmented_size).astype(bool)
        self.state_len = len(self.use)
        # Update estimated satellite
        self.est_sat.match_estimate(est_state=self.x_hat, dt=dt)

    def _from_augmented_array(
        self,
        value: np.ndarray,
        *,
        cov: np.ndarray | None = None,
        int_cov: np.ndarray | None = None,
    ) -> EstimatorState:
        return EstimatorState.from_estimator_array(
            value,
            n_rw=self.est_sat.number_RW,
            n_act_bias=self.est_sat.act_bias_len,
            n_sens_bias=self.est_sat.att_sens_bias_len,
            n_dist_param=self.est_sat.dist_param_len,
            cov=self.x_hat.cov if cov is None else cov,
            int_cov=self.x_hat.int_cov if int_cov is None else int_cov,
        )


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

        This routine builds an initial attitude quaternion (and optionally a
        gyro bias / angular rate guess) from a subset of available sensor
        measurements. Any measurement may be omitted by passing ``None``.

        Attitude initialization via Wahba's problem
        -------------------------------------------
        If an initial quaternion ``q`` is not provided, this method estimates it
        by solving Wahba's problem using all available vector observations.

        Given body-frame measurements :math:`\mathbf{v}_{b,i}` and their inertial
        references :math:`\mathbf{v}_{I,i}`, we seek the rotation :math:`\mathbf{R}`
        (or quaternion :math:`\mathbf{q}`) minimizing

        .. math::

            J(\mathbf{R}) = \sum_{i=1}^{m} w_i \left\lVert \mathbf{v}_{b,i} - \mathbf{R}\,\mathbf{v}_{I,i} \right\rVert^2,

        with weights :math:`w_i>0`. The solution is computed using
        :func:`~ADCS.helpers.math_helpers.wahbas_svd`.

        Measurement weighting
        ---------------------
        Each available vector observation is weighted by an inverse-variance-like
        term combining sensor noise and current prior bias uncertainty:

        .. math::

            w_i = \frac{1}{\sigma_i^2 + P_{\text{bias},i}},

        where :math:`\sigma_i^2` is the sensor noise variance and
        :math:`P_{\text{bias},i}` is the corresponding scalar bias variance
        extracted from the current covariance :attr:`x_hat.cov` at the sensor's
        bias index.

        Sensors used (when available) are:

        - Magnetometer (body) with reference :math:`\mathbf{B}_{ECI}`,
        - Sun sensor (body) with reference :math:`\mathbf{S}_{ECI}`,
        - Sun sensor pair (body) with reference :math:`\mathbf{S}_{ECI}`.

        Gyro bias and angular-rate guess (optional)
        -------------------------------------------
        If a gyroscope reading :math:`\tilde{\boldsymbol{\omega}}` is available,
        the method forms a simple linear estimator using:

        - gyro bias covariance block :math:`P_b`,
        - gyro measurement noise :math:`R_g`,
        - rate covariance :math:`Q_\omega`.

        A typical structure consistent with the implementation is

        .. math::

            \hat{\mathbf{b}}_g
            = (R_g + P_b + Q_\omega)^{-1} P_b \,\tilde{\boldsymbol{\omega}}.

        The method then constructs an approximate rate guess
        :math:`\hat{\boldsymbol{\omega}}` from the bias and covariance blocks.

        Satellite synchronization
        -------------------------
        The updated augmented state is committed into :attr:`x_hat.as_estimator_array()` and then
        pushed to the internal satellite model via
        :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

        :param gyro_readings: Gyroscope measurement in the body frame, shape ``(3,)``.
            If ``None``, gyro bias and rate are not updated.
        :type gyro_readings: numpy.ndarray or None
        :param mtm_readings: Magnetometer measurement in the body frame, shape ``(3,)``.
        :type mtm_readings: numpy.ndarray or None
        :param sunpair_readings: Sun sensor pair measurement in the body frame, shape ``(3,)``.
        :type sunpair_readings: numpy.ndarray or None
        :param sunsensor_readings: Sun sensor measurement in the body frame, shape ``(3,)``.
        :type sunsensor_readings: numpy.ndarray or None
        :param B_ECI: Reference magnetic field vector in ECI (or inertial) frame, shape ``(3,)``.
        :type B_ECI: numpy.ndarray or None
        :param S_ECI: Reference sun direction vector in ECI (or inertial) frame, shape ``(3,)``.
        :type S_ECI: numpy.ndarray or None
        :param os: Current orbital state (included for interface consistency).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param q: Optional initial quaternion, shape ``(4,)``. If provided, Wahba's problem is skipped.
        :type q: numpy.ndarray or None
        :raises RuntimeError: If no valid vector measurements are available for Wahba initialization.
        :return: Updated full augmented state vector (same length as :attr:`x_hat.as_estimator_array()`).
        :rtype: numpy.ndarray

        """

        # ---- 1. Build lists of available vectors for Wahba ----------------------

        body_vecs = []
        eci_vecs  = []
        weights   = []

        # ---- Magnetometer ----
        if mtm_readings is not None and B_ECI is not None:
            sens = self.est_sat.mtm_sensor   # You will have this reference
            var = sens.noise.std_noise**2 #* sens.scale**2

            bias_idx = sens.bias_state_index  # precomputed during satellite setup
            P_bias = self.x_hat.cov[bias_idx, bias_idx]

            body_vecs.append(mtm_readings)
            eci_vecs.append(B_ECI)
            weights.append(1.0 / (var + P_bias))

        # ---- Sun sensor (simple) ----
        if sunsensor_readings is not None and S_ECI is not None:
            sens = self.est_sat.sun_sensor
            var  = sens.noise.std_noise**2 #* sens.scale**2
            bias_idx = sens.bias_state_index
            P_bias = self.x_hat.cov[bias_idx, bias_idx]

            body_vecs.append(sunsensor_readings)
            eci_vecs.append(S_ECI)
            weights.append(1.0 / (var + P_bias))

        # ---- Sun sensor pair ----
        if sunpair_readings is not None and S_ECI is not None:
            sens = self.est_sat.sunpair_sensor
            var  = sens.noise.std_noise**2 #* sens.scale**2
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
        state = self.x_hat.as_estimator_array()
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
        self.x_hat = self._from_augmented_array(state)
        self.est_sat.match_estimate(self.x_hat, dt=1.0)

        return state


    def update(self, u: np.ndarray, sensors: Sequence[np.ndarray], os: Orbital_State) -> EstimatorState:
        r"""
        High-level estimator update wrapper.

        This method delegates the actual filtering step to a subclass-defined
        :meth:`update_core`, optionally enforces a block-structured covariance,
        updates the stored estimate :attr:`x_hat`, and synchronizes the satellite
        model via :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

        Delegation to subclass filter
        -----------------------------
        Subclasses must implement ``update_core`` and return an
        :class:`~ADCS.state.EstimatorState`
        containing:

        - full augmented state :math:`\hat{\mathbf{x}}`,
        - ``cov``: covariance in reduced coordinates (unless ``quat_as_vec=True``),
        - ``int_cov``: process-noise covariance (often unchanged).

        Previous orbital state handling
        -------------------------------
        If :attr:`prev_os` has not been initialized (``prev_os.J2000 == 0``),
        it is set to the provided :class:`~ADCS.orbits.orbital_state.Orbital_State`
        ``os`` on the first call.

        Optional covariance block-zeroing
        ----------------------------------
        If :attr:`cross_term` is ``False``, this method zeros cross-covariance
        blocks between selected groups to enforce a block structure. Let the
        reduced covariance be partitioned as

        .. math::

            P =
            \begin{bmatrix}
                P_{xx} & P_{x,a} & P_{x,s} & P_{x,d} \\
                P_{a,x} & P_{aa} & P_{a,s} & P_{a,d} \\
                P_{s,x} & P_{s,a} & P_{ss} & P_{s,d} \\
                P_{d,x} & P_{d,a} & P_{d,s} & P_{dd}
            \end{bmatrix},

        where:

        - ``a`` denotes actuator-bias states,
        - ``s`` denotes attitude-sensor-bias states,
        - ``d`` denotes disturbance parameters.

        Then the enforced structure is:

        .. math::

            P_{a,s}=P_{s,a}^\top=\mathbf{0},\quad
            P_{a,d}=P_{d,a}^\top=\mathbf{0},\quad
            P_{s,d}=P_{d,s}^\top=\mathbf{0}.

        The index ranges are derived from:

        - :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.state_len`,
        - :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.act_bias_len`,
        - :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.att_sens_bias_len`,
        - :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.dist_param_len`.

        Committing the estimate
        -----------------------
        The new estimate is written back into :attr:`x_hat` using a mask over the
        reduced covariance space returned by
        :meth:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator.cov_use`.
        Subselection and reinsertion of covariance/process-noise blocks relies on
        :func:`~ADCS.helpers.math_helpers.square_mat_sections`.

        :param u: Control input vector applied to the satellite (e.g., RW commands).
        :type u: numpy.ndarray
        :param sensors: Collection of sensor measurement vectors used by the estimator.
            The stacking/order convention is defined by the subclass and the satellite model.
        :type sensors: list[numpy.ndarray]
        :param os: Current orbital/environmental state used for propagation and prediction.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Updated estimator state container.
        :rtype: :class:`~ADCS.state.EstimatorState`

        """
        if self.prev_os.J2000 == 0:
            self.prev_os = os

        x_hat: EstimatorState = self.update_core(u=u, sensors=sensors, os=os)

        self.prev_os = os
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

        current = self.x_hat.as_estimator_array()
        updated = x_hat.as_estimator_array()
        current[self.use] = updated[self.use]
        cov_current = self.x_hat.cov.copy()
        cov_mask = self.cov_use()
        cov_current[np.ix_(cov_mask, cov_mask)] = oc
        self.x_hat = self._from_augmented_array(
            current,
            cov=cov_current,
            int_cov=self.x_hat.int_cov,
        )
        self.est_sat.match_estimate(self.x_hat, self.dt)

        return self.x_hat.copy()


    def cov_use(self) -> np.ndarray:
        r"""
        Return a boolean mask for the reduced covariance state.

        The attribute :attr:`use` is a boolean mask over the **full** augmented
        state :math:`\mathbf{x}\in\mathbb{R}^{N}`. The covariance is stored in a
        reduced attitude-error space of dimension :math:`N-1`, where the quaternion
        is represented minimally.

        This method converts the full-state mask into a reduced-state mask by
        deleting the entry corresponding to the dropped quaternion component
        (index 3 in the full state vector ordering).

        Quaternion reduction convention
        --------------------------------
        If the full state begins as

        .. math::

            \mathbf{x} =
            \begin{bmatrix}
                \omega_x \\ \omega_y \\ \omega_z \\
                q_0 \\ q_1 \\ q_2 \\ q_3 \\
                \vdots
            \end{bmatrix},

        then index 3 corresponds to the first quaternion component in the full
        storage convention and is removed to obtain a mask compatible with the
        reduced covariance representation.

        :param: None.
        :type: None
        :return: Boolean mask of length :math:`N-1` compatible with the reduced covariance matrices.
        :rtype: numpy.ndarray

        """
        if self.quat_as_vec:
            return self.use.copy()
        return np.delete(self.use, 3)
