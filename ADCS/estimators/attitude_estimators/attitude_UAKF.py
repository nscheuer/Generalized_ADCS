__all__ = ["UAKF"]

import numpy as np
import copy
import scipy
from typing import List
import time

from ADCS.estimators.attitude_estimators.attitude_estimator import Attitude_Estimator
from ADCS.state import EstimatorState, State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import CG5
from ADCS.helpers.math_helpers import (
    quat_to_vec3,
    vec3_to_quat,
    quat_mult,
    quat_inv,
    normalize,
    state_norm_jac,
    matrix_row_normalize,
)


class UAKF(Attitude_Estimator):
    r"""
    Unscented Kalman Filter (UKF) with an error-state attitude representation.

    This class implements an Unscented Kalman Filter (UKF) for spacecraft attitude
    estimation and associated bias/parameter states. It inherits the augmented-state
    bookkeeping (full-state storage, reduced covariance representation, and satellite
    synchronization) from :class:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator`.

    The implementation uses an **error-state attitude representation** by default
    (``quat_as_vec=False``), in which the quaternion is stored in the full state
    vector but the covariance is stored in a minimal 3-parameter attitude-error
    coordinate (e.g., :math:`\delta\boldsymbol{\theta}\in\mathbb{R}^3`). If
    ``quat_as_vec=True``, the quaternion is included as a 4-vector in the covariance.

    State and covariance structure
    ------------------------------
    Let the full augmented state be

    .. math::

        \mathbf{x} =
        \begin{bmatrix}
            \boldsymbol{\omega} \\
            \mathbf{q} \\
            \mathbf{z}
        \end{bmatrix},

    where :math:`\boldsymbol{\omega}\in\mathbb{R}^3` is the angular rate,
    :math:`\mathbf{q}\in\mathbb{R}^4` is the unit quaternion, and :math:`\mathbf{z}`
    stacks remaining physical/bias/disturbance states.

    In reduced error-state mode, the filter internally works with an error vector

    .. math::

        \delta\mathbf{x} =
        \begin{bmatrix}
            \delta\boldsymbol{\omega} \\
            \delta\boldsymbol{\theta} \\
            \delta\mathbf{z}
        \end{bmatrix},

    with :math:`\delta\boldsymbol{\theta}\in\mathbb{R}^3` representing attitude error,
    typically mapped to a quaternion increment :math:`\delta\mathbf{q}` and composed
    multiplicatively with a reference quaternion. The full state always stores
    :math:`\mathbf{q}` explicitly.

    Unscented transform overview
    ----------------------------
    For an augmented dimension :math:`L`, the UKF constructs sigma points
    :math:`\chi_i` around the current mean using

    .. math::

        \lambda = \alpha^2 (L + \kappa) - L,\qquad
        \gamma = \sqrt{L + \lambda}.

    Mean and covariance weights are

    .. math::

        W_0^{(m)} = \frac{\lambda}{L+\lambda},\qquad
        W_0^{(c)} = \frac{\lambda}{L+\lambda} + (1-\alpha^2+\beta),

    .. math::

        W_i^{(m)} = W_i^{(c)} = \frac{1}{2(L+\lambda)},\quad i=1,\ldots,2L.

    Sigma points are propagated through the nonlinear dynamics

    .. math::

        \chi_i^- = f(\chi_i, \mathbf{u}, \mathbf{w}_i),

    where propagation is performed by
    :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.noiseless_rk4`
    with appropriate injected control perturbations. Predicted measurements are

    .. math::

        \gamma_i^- = h(\chi_i^-),

    computed using
    :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.sensor_readings`
    (with an :class:`~ADCS.satellite_hardware.errors.helpers.error_mode.ErrorMode` selecting bias/noise behavior).

    Predicted means are

    .. math::

        \bar{\mathbf{x}}^- = \sum_{i=0}^{2L} W_i^{(m)} \chi_i^-,\qquad
        \bar{\mathbf{y}}^- = \sum_{i=0}^{2L} W_i^{(m)} \gamma_i^-.

    Covariances are

    .. math::

        P^- = \sum_{i=0}^{2L} W_i^{(c)}(\chi_i^- - \bar{\mathbf{x}}^-)(\chi_i^- - \bar{\mathbf{x}}^-)^\top + Q,

    .. math::

        P_{yy} = \sum_{i=0}^{2L} W_i^{(c)}(\gamma_i^- - \bar{\mathbf{y}}^-)(\gamma_i^- - \bar{\mathbf{y}}^-)^\top + R,

    .. math::

        P_{y x} = \sum_{i=0}^{2L} W_i^{(c)}(\gamma_i^- - \bar{\mathbf{y}}^-)(\chi_i^- - \bar{\mathbf{x}}^-)^\top.

    The gain is formed as

    .. math::

        K = P_{x y} P_{yy}^{-1},

    and the mean/covariance are updated using innovation
    :math:`\boldsymbol{\nu} = \mathbf{y} - \bar{\mathbf{y}}^-`:

    .. math::

        \bar{\mathbf{x}}^+ = \bar{\mathbf{x}}^- + K\boldsymbol{\nu},\qquad
        P^+ = P^- - K P_{yy} K^\top.

    Quaternion error composition
    ----------------------------
    In reduced attitude-error mode, attitude is updated multiplicatively:

    .. math::

        \delta\mathbf{q} = \operatorname{vec3\_to\_quat}(\delta\boldsymbol{\theta}),\qquad
        \mathbf{q}^+ = \mathbf{q}^- \otimes \delta\mathbf{q},

    using :func:`~ADCS.helpers.math_helpers.vec3_to_quat` and
    :func:`~ADCS.helpers.math_helpers.quat_mult`. If ``quat_as_vec=True``,
    the quaternion is renormalized via :func:`~ADCS.helpers.math_helpers.normalize`,
    and covariance is transformed using :func:`~ADCS.helpers.math_helpers.state_norm_jac`.

    """
    def __init__(
        self,
        est_sat: EstimatedSatellite,
        J2000: float,
        x_hat: np.ndarray,
        P_hat: np.ndarray,
        Q_hat: np.ndarray,
        dt: float = 1.0,
        cross_term: bool = False,
        quat_as_vec: bool = False,
        ut_alpha: float = 1e-3,
    ) -> None:
        r"""
        Initialize the UKF estimator and store UKF scaling parameters.

        This constructor initializes the base estimator bookkeeping via
        :class:`~ADCS.estimators.attitude_estimators.attitude_estimator.Attitude_Estimator`,
        then sets UKF tuning parameters:

        .. math::

            \alpha,\ \beta,\ \kappa

        stored as :attr:`al`, :attr:`bet`, and :attr:`kap`, and selects the
        3-vector attitude parameterization mode used by
        :func:`~ADCS.helpers.math_helpers.quat_to_vec3` and
        :func:`~ADCS.helpers.math_helpers.vec3_to_quat`.

        :param est_sat: Estimated satellite model defining state structure and dynamics.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param J2000: Initial time in seconds since J2000.
        :type J2000: float
        :param x_hat: Initial full augmented state estimate.
        :type x_hat: ADCS.state.EstimatorState
        :param P_hat: Initial covariance matrix (reduced error-state unless ``quat_as_vec=True``).
        :type P_hat: numpy.ndarray
        :param Q_hat: Initial process-noise covariance matrix (same convention as ``P_hat``).
        :type Q_hat: numpy.ndarray
        :param dt: Propagation time step :math:`\Delta t` in seconds.
        :type dt: float
        :param cross_term: Whether cross-covariances between bias/parameter blocks are preserved.
        :type cross_term: bool
        :param quat_as_vec: If ``True``, include quaternion as a 4-vector in covariance; otherwise use 3-DOF attitude error.
        :type quat_as_vec: bool
        :return: ``None``.
        :rtype: None

        """
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
            quat_as_vec=quat_as_vec,
        )
        # UKF tuning parameters.
        #
        # alpha controls the unscented sigma-point spread:
        #   lambda = alpha^2 (L + kappa) - L,   gamma = sqrt(L + lambda).
        #
        # alpha = 1e-3 (the textbook "default") gives gamma ~ alpha*sqrt(L)
        # ~ 2.4e-3 with covariance weights ~ +/-1e6 -- the UT degenerates to a
        # local (EKF-grade) linearisation and *under*-estimates the attitude
        # covariance (attitude NEES ~263 vs consistent 3). alpha = 1 gives
        # gamma = sqrt(L), O(1) weights, ~5x better attitude NEES (~55).
        #
        # BUT alpha = 1 is a CONFIG-DEPENDENT TRADEOFF, not a free win: it
        # deterministically REGRESSES reaction-wheel-momentum estimation in
        # the RWS UKF/SRUKF variants (test_rw_momentum_estimation diverges;
        # origin/main alpha=1e-3 passes, alpha=1 fails). It is therefore
        # exposed as an OPT-IN parameter (`ut_alpha`) with the LEGACY default
        # 1e-3 -> zero behavioural regression. Opt in to alpha=1 only where
        # the augmented config is known safe (e.g. the attitude NEES guard
        # tests). The principled resolution (derive RW-momentum / all
        # augmented-state process noise so the filter is well-posed
        # independent of alpha) is documented future work
        # (~/adcs-srukf-process-noise-notes.md sections 7i/7j/7k).
        self.al = ut_alpha
        self.kap = 0.0
        self.bet = 2.0
        self.vec_mode = 6

    def determine_covariances_to_use(self, state_cov: np.ndarray, sens_cov: np.ndarray, control_cov: np.ndarray, int_cov: np.ndarray) -> List[bool]:
        r"""
        Determine which covariance blocks are included in sigma-point augmentation.

        The UKF may be implemented with an **augmented state** that includes
        process noise and/or measurement noise as additional random variables.
        This method encodes the estimator's policy for which covariance blocks
        contribute to the augmented dimension :math:`L` used in sigma point
        generation.

        In this implementation, the returned list corresponds to:

        .. math::

            [\text{state},\ \text{sensor},\ \text{control},\ \text{process}],

        determining whether each covariance block is included in the sigma-point
        construction in :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.make_pts_and_wts`.

        :param state_cov: State covariance matrix :math:`P` (reduced or full according to ``quat_as_vec``).
        :type state_cov: numpy.ndarray
        :param sens_cov: Sensor noise covariance matrix :math:`R` for active sensors.
        :type sens_cov: numpy.ndarray
        :param control_cov: Control noise covariance associated with actuation commands.
        :type control_cov: numpy.ndarray
        :param int_cov: Process (integration) noise covariance :math:`Q`.
        :type int_cov: numpy.ndarray
        :return: A 4-element boolean list indicating whether to include
            ``[state, sensor, control, process]`` covariance blocks.
        :rtype: list[bool]

        """
        use_state_cov = True
        use_sens_cov = False
        use_control_cov = np.size(control_cov)>0 and not np.all(control_cov==0)
        use_int_cov = False
        return [use_state_cov, use_sens_cov, use_control_cov, use_int_cov]

    # ------------------------------------------------------------------ #
    # Sensor mask expansion
    # ------------------------------------------------------------------ #
    def _expand_sensor_mask(self, which_sensors: List[bool]) -> np.ndarray:
        r"""
        Expand a per-sensor activity mask into a per-output activity mask.

        Some attitude sensors produce multi-dimensional outputs (e.g., a 3-vector),
        while others produce scalar outputs. The measurement stack used by the
        filter is formed by concatenating outputs. This method expands a boolean
        activity mask defined per sensor into a boolean mask defined per output
        dimension, suitable for indexing stacked measurement vectors.

        Let sensor :math:`j` have output length :math:`d_j`. Given a mask
        :math:`m_j\in\{\text{True},\text{False}\}`, this method produces an output-level
        mask with :math:`d_j` repeated entries:

        .. math::

            \mathbf{m}_{out} =
            [\underbrace{m_1,\ldots,m_1}_{d_1},
             \underbrace{m_2,\ldots,m_2}_{d_2},
             \ldots].

        Reaction wheel actuator "sensors" are appended as scalar outputs
        (:math:`d=1` each) following the attitude sensors.

        :param which_sensors: Boolean mask indicating which sensors are active (one bool per sensor entry).
        :type which_sensors: list[bool]
        :return: Boolean mask with one element per stacked measurement output dimension.
        :rtype: numpy.ndarray

        """
        mask = []
        for j, sensor in enumerate(self.est_sat.attitude_sensors):
            output_len = sensor.output_length
            mask.extend([which_sensors[j]] * output_len)
        # Add RW sensors (each has output_length=1)
        for k, rw in enumerate(self.est_sat.rw_actuators):
            sensor_idx = len(self.est_sat.attitude_sensors) + k
            mask.append(which_sensors[sensor_idx] if sensor_idx < len(which_sensors) else True)
        return np.array(mask, dtype=bool)


    def make_pts_and_wts(self, pt0: np.ndarray, which_sensors: List[bool]):
        r"""
        Build augmented sigma points and Unscented Transform weights.

        This method constructs sigma points in an augmented space comprised of
        selected covariance blocks (state, sensor noise, control noise, process noise).
        The inclusion policy is determined by
        :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.determine_covariances_to_use`.

        Augmented dimension
        -------------------
        Let covariance blocks be:

        - state covariance :math:`P_x`,
        - sensor covariance :math:`R`,
        - control covariance :math:`Q_u`,
        - process covariance :math:`Q`.

        With inclusion flags :math:`b_j\in\{0,1\}`, the augmented dimension is

        .. math::

            L = \sum_{j} b_j\, n_j,

        where :math:`n_j` is the dimension of the corresponding block.

        Sigma point generation
        ----------------------
        Define

        .. math::

            \lambda = \alpha^2(L+\kappa) - L,\qquad
            \gamma = \sqrt{L + \lambda},\qquad
            \text{scale} = L + \lambda.

        For each included block with covariance :math:`P`, the method forms the
        scaled Cholesky factor:

        .. math::

            \text{scale}\,P = LL^\top,

        using :func:`numpy.linalg.cholesky`, then creates offsets

        .. math::

            \Delta_i \in \{\pm L_{:,k}\}_{k=1}^{n},

        yielding :math:`2n` offsets per block.

        For the **state block** (error-state), offsets are mapped onto the manifold
        using :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.add_to_state`,
        ensuring multiplicative quaternion perturbations when ``quat_as_vec=False``.

        For non-state blocks (noise blocks), offsets are inserted into the sigma-point
        container list structure as additive perturbations on those blocks only.

        Weights
        -------
        Mean and covariance weights follow the standard UKF formulas:

        .. math::

            W_0^{(m)} = \frac{\lambda}{L+\lambda},\qquad
            W_0^{(c)} = \frac{\lambda}{L+\lambda} + (1-\alpha^2+\beta),

        .. math::

            W_i^{(m)} = W_i^{(c)} = \frac{1}{2(L+\lambda)},\quad i=1,\ldots,2L.

        Output formats
        --------------
        The method returns:

        - ``pts``: a list of length :math:`2L+1` where each entry is:

          .. math::

              [\text{full\_state},\ \text{sens\_noise},\ \text{control\_noise},\ \text{int\_noise}].

        - ``sig0``: a stacked array of the state component only, aligned with ``pts``.
          Points not perturbed in the state block are padded with the mean state.

        :param pt0: Current mean full state vector.
        :type pt0: numpy.ndarray
        :param which_sensors: Boolean mask indicating which sensors are active/valid.
        :type which_sensors: list[bool]
        :return: Tuple ``(L, pts, wts_m, wts_c, sig0)`` where:

            - ``L`` is the augmented dimension,
            - ``pts`` is the sigma-point container list,
            - ``wts_m`` are mean weights (length :math:`2L+1`),
            - ``wts_c`` are covariance weights (length :math:`2L+1`),
            - ``sig0`` is the stacked state-only sigma-point array aligned with ``pts``.

        :rtype: tuple[int, list, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        # Copy covariances (same shapes/values as original)
        state_cov = self.x_hat.cov.copy()
        int_cov = self.x_hat.int_cov.copy() * 0.0
        control_cov = self.est_sat.control_cov()
        sens_cov = self.est_sat.sensor_cov(which_sensors=which_sensors) * 0.0
        
        include_cov = self.determine_covariances_to_use(state_cov, sens_cov, control_cov, int_cov)
        covs = [state_cov, sens_cov, control_cov, int_cov]

        zeros_state = pt0
        zeros_sens = sens_cov[0, :] * 0.0 if sens_cov.size else np.zeros(0, dtype=pt0.dtype)
        zeros_control = control_cov[0, :] * 0.0 if control_cov.size else np.zeros(0, dtype=pt0.dtype)
        zeros_int = int_cov[0, :] * 0.0 if int_cov.size else np.zeros(0, dtype=pt0.dtype)

        zeros = [zeros_state, zeros_sens, zeros_control, zeros_int]

        # Total augmented dimension (identical formula)
        L = int(sum(include_cov[j] * covs[j].shape[0] for j in range(4)))

        lam = self.al ** 2.0 * (self.kap + L) - L
        self.scale = L + lam

        pts = [zeros]
        offsets: List[np.ndarray] = [None, None, None, None]  # type: ignore[assignment]

        # Generate sigma-point offsets per block
        for j in range(4):
            if not include_cov[j]:
                continue

            cov_j = covs[j]
            if cov_j.size == 0:
                continue

            # Cholesky of scaled covariance
            chol_mat = self.scale * cov_j
            if j == 0:
                # State covariance must be positive definite; let a failure
                # propagate rather than masking a diverged filter.
                mat = np.linalg.cholesky(chol_mat)  # (dim, dim)
            else:
                # Noise blocks may be PSD-singular (e.g. rank-deficient
                # actuator noise); mirror the SRUAKF eigh fallback instead of
                # aborting the update.
                try:
                    mat = np.linalg.cholesky(chol_mat)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(chol_mat)
                    mat = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
            # 2*dim sigma offsets: +columns and -columns
            offs = np.hstack((mat, -mat)).T  # (2*dim, dim)
            offsets[j] = offs

            if j == 0:
                # Error-state -> full-state sigma points (same as original)
                states = self.add_to_state(pt0, offs)
                pts.extend(
                    [zeros[:j] + [k] + zeros[j + 1 :] for k in states]
                )
            else:
                pts.extend(
                    [zeros[:j] + [k] + zeros[j + 1 :] for k in offs]
                )

        # Weights (same formulas)
        denom = L + lam
        w0_m = lam / denom
        w0_c = lam / denom + (1.0 - self.al ** 2.0 + self.bet)
        wi = 0.5 / denom

        num_sigma = 2 * L + 1
        wts_m = np.empty(num_sigma, dtype=float)
        wts_c = np.empty(num_sigma, dtype=float)
        wts_m[0] = w0_m
        wts_c[0] = w0_c
        wts_m[1:] = wi
        wts_c[1:] = wi

        self.wts_m = wts_m
        self.wts_c = wts_c

        # sig0: first-block state sigma points padded to 2L+1 with pt0
        # (matches original np.vstack([pt0, states] + [pt0]*...))
        if offsets[0] is not None:
            states = self.add_to_state(pt0, offsets[0])
            pad_count = num_sigma - (1 + states.shape[0])
            if pad_count > 0:
                sig0 = np.vstack(
                    (pt0, states, np.repeat(pt0[None, :], pad_count, axis=0))
                )
            else:
                sig0 = np.vstack((pt0, states))
        else:
            sig0 = np.repeat(pt0[None, :], num_sigma, axis=0)

        return L, pts, wts_m, wts_c, sig0


    def reunite_states(
        self,
        dynstate: np.ndarray,
        rest_state: np.ndarray,
        quatref: np.ndarray,
    ) -> np.ndarray:
        r"""
        Reassemble the reduced error-state vector from propagated dynamics and static components.

        This helper constructs a reduced state representation used internally by
        the filter for sigma-point bookkeeping.

        Quaternion mapping
        ------------------
        If ``quat_as_vec=True``, the quaternion is treated as part of the additive
        state vector and the method simply concatenates:

        .. math::

            \mathbf{x}_{red} = \begin{bmatrix}\mathbf{x}_{dyn} \\ \mathbf{x}_{rest}\end{bmatrix}.

        If ``quat_as_vec=False``, the propagated dynamic state ``dynstate`` contains
        the full quaternion :math:`\mathbf{q}`. The method converts this quaternion
        into a 3-vector attitude error relative to a reference quaternion
        :math:`\mathbf{q}_{ref}` by computing:

        .. math::

            \delta\mathbf{q} = \mathbf{q}_{ref}^{-1} \otimes \mathbf{q},

        using :func:`~ADCS.helpers.math_helpers.quat_inv` and
        :func:`~ADCS.helpers.math_helpers.quat_mult`, then mapping
        :math:`\delta\mathbf{q}` to a 3-vector:

        .. math::

            \delta\boldsymbol{\theta} = \operatorname{quat\_to\_vec3}(\delta\mathbf{q}),

        via :func:`~ADCS.helpers.math_helpers.quat_to_vec3`.

        The returned reduced vector is arranged as:

        .. math::

            [\boldsymbol{\omega},\ \delta\boldsymbol{\theta},\ \text{(other dyn)},\ \text{rest\_state}].

        :param dynstate: Dynamic part of the state (rate + attitude + other dynamic entries); contains a full quaternion.
        :type dynstate: numpy.ndarray
        :param rest_state: Static/bias/parameter portion of the state appended after the dynamic portion.
        :type rest_state: numpy.ndarray
        :param quatref: Reference quaternion used to compute the attitude error (ignored if ``quat_as_vec=True``).
        :type quatref: numpy.ndarray
        :return: Reduced error-state vector consistent with the estimator's internal representation.
        :rtype: numpy.ndarray

        """
        if self.quat_as_vec:
            return np.concatenate((dynstate, rest_state))
        else:
            # dynstate has full attitude quaternion already
            quatdiff = quat_mult(quat_inv(quatref), dynstate[3:7])
            v3diff = quat_to_vec3(quatdiff, self.vec_mode)
            return np.concatenate((dynstate[0:3], v3diff, dynstate[7:], rest_state))

    def add_to_state(self, state: np.ndarray, add: np.ndarray) -> np.ndarray:
        r"""
        Apply an error perturbation to a base state on the attitude manifold.

        This method implements the sigma-point "addition" operator. In Euclidean
        state components, it is additive. For quaternion attitude, it is
        **multiplicative** when ``quat_as_vec=False``.

        Single-point (vector) case
        --------------------------
        For a single base state :math:`\mathbf{x}` and perturbation
        :math:`\delta\mathbf{x}`, the update is:

        - If ``quat_as_vec=True`` (additive quaternion with renormalization):

          .. math::

              \mathbf{x}^+ = \mathbf{x} + \delta\mathbf{x},\qquad
              \mathbf{q}^+ \leftarrow \frac{\mathbf{q}^+}{\lVert \mathbf{q}^+ \rVert}.

          Quaternion normalization uses :func:`~ADCS.helpers.math_helpers.normalize`.

        - If ``quat_as_vec=False`` (multiplicative quaternion):

          .. math::

              \boldsymbol{\omega}^+ = \boldsymbol{\omega} + \delta\boldsymbol{\omega},

          .. math::

              \delta\mathbf{q} = \operatorname{vec3\_to\_quat}(\delta\boldsymbol{\theta}),\qquad
              \mathbf{q}^+ = \mathbf{q} \otimes \delta\mathbf{q},

          .. math::

              \mathbf{z}^+ = \mathbf{z} + \delta\mathbf{z}.

          Quaternion composition uses :func:`~ADCS.helpers.math_helpers.quat_mult`
          and the vector-to-quaternion map uses
          :func:`~ADCS.helpers.math_helpers.vec3_to_quat`.

        Batch (matrix) case
        -------------------
        If ``add`` is a 2D array (sigma-point perturbations), the same operations
        are applied row-wise. For ``quat_as_vec=True``, quaternion rows are
        normalized with :func:`~ADCS.helpers.math_helpers.matrix_row_normalize`.

        :param state: Base state vector (1D) or base full state vector (1D) used for all sigma points.
        :type state: numpy.ndarray
        :param add: Perturbation/error vector(s), shape ``(n,)`` or ``(k,n)``.
        :type add: numpy.ndarray
        :return: Updated state vector(s) after applying the perturbation on the manifold.
        :rtype: numpy.ndarray

        """
        add = np.squeeze(add)
        state = np.squeeze(state)

        if add.ndim == 1:
            if self.quat_as_vec:
                result = state + add
                result[3:7] = normalize(result[3:7])
            else:
                result = state.copy()
                result[0:3] = state[0:3] + add[0:3]
                result[7:] = state[7:] + add[6:]
                result[3:7] = quat_mult(
                    state[3:7],
                    vec3_to_quat(add[3:6], self.vec_mode),
                )
        else:
            if self.quat_as_vec:
                result = state + add
                result[:, 3:7] = matrix_row_normalize(result[:, 3:7])
            else:
                n_sigma = add.shape[0]
                n_state = state.shape[0]
                result = np.empty((n_sigma, n_state), dtype=state.dtype)
                result[:, 0:3] = state[0:3] + add[:, 0:3]
                result[:, 7:] = state[7:] + add[:, 6:]
                # quaternion update per sigma point (still loop, same math)
                quats = [
                    quat_mult(
                        state[3:7],
                        vec3_to_quat(add[j, 3:6], self.vec_mode),
                    )
                    for j in range(n_sigma)
                ]
                result[:, 3:7] = np.vstack(quats)
        return result

    def new_post_state(
        self,
        pre_rest_state: np.ndarray,
        post_dynstate: np.ndarray,
        int_err: np.ndarray,
        quatref: np.ndarray,
    ):
        r"""
        Construct posterior reduced and full states after propagation with integration error.

        This helper reconstructs, for a given sigma point, both:

        - ``post_state``: the reduced error-state representation used internally by the UKF,
        - ``full_state``: the full augmented state with quaternion suitable for satellite models.

        Integration error injection
        ---------------------------
        The integration error vector ``int_err`` is interpreted in the reduced
        error-state coordinates and partitioned into:

        - a head portion applied to the dynamic state on the manifold via
          :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.add_to_state`,
        - a tail portion applied additively to the remaining bias/parameter states.

        If ``state_len`` is the satellite dynamic-state length (including quaternion),
        the reduced "head length" used is

        .. math::

            n_h = \text{state\_len} - 1 + \mathbf{1}[\text{quat\_as\_vec}],

        matching the implementation.

        Full-state reconstruction
        -------------------------
        After obtaining the reduced posterior vector ``post_state``, the method
        constructs an auxiliary reference state containing ``quatref`` and then
        re-applies the reduced perturbation using
        :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.add_to_state`
        to produce a full state with a quaternion.

        :param pre_rest_state: Bias/parameter states before propagation (assumed constant through dynamics).
        :type pre_rest_state: numpy.ndarray
        :param post_dynstate: Propagated dynamic state (rate + quaternion + other dynamic entries).
        :type post_dynstate: numpy.ndarray
        :param int_err: Integration error/process-noise vector for this sigma point (reduced coordinates).
        :type int_err: numpy.ndarray
        :param quatref: Reference quaternion used for sigma-point attitude error mapping.
        :type quatref: numpy.ndarray
        :return: Tuple ``(post_state, full_state)`` where ``post_state`` is the reduced
            error-state vector and ``full_state`` is the full augmented state vector.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        """
        # integration error is in reduced error-state coordinates
        state_len = self.est_sat.state_len
        head_len = state_len - 1 + self.quat_as_vec

        post_dyn_state_w_int_err = self.add_to_state(
            post_dynstate, int_err[:head_len]
        )
        post_state = self.reunite_states(
            post_dyn_state_w_int_err,
            pre_rest_state + int_err[head_len:],
            quatref,
        )
        s0len = np.zeros(post_state.size + 1 - self.quat_as_vec, dtype=post_state.dtype)
        s0len[3:7] = quatref
        # These are "backwards" on purpose (same as original code)
        full_state = self.add_to_state(s0len, post_state)
        return post_state, full_state


    def sat_match(self, est_sat: EstimatedSatellite, state: EstimatorState) -> None:
        r"""
        Synchronize an :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        instance with an estimator state.

        Sigma-point propagation and measurement prediction require the satellite model
        to reflect the sigma point's specific biases/parameters and attitude. This method
        applies the provided :class:`~ADCS.state.EstimatorState` to the satellite model via
        :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

        :param est_sat: Satellite model instance to be updated for sigma-point propagation.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param state: Estimator state to apply.
        :type state: :class:`~ADCS.state.EstimatorState`
        :return: ``None``.
        :rtype: None

        """
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        est_sat.match_estimate(state, self.dt)


    def update_core(
        self,
        u: np.ndarray,
        sensors: np.ndarray,   # was effectively used as a NumPy array already
        os: Orbital_State,
    ) -> EstimatorState:
        r"""
        Perform one UKF predict/update cycle in reduced error-state coordinates.

        This method implements a complete UKF step:

        1. Determine active sensors from the provided measurement vector (NaN check),
           then expand to an output-level mask via
           :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF._expand_sensor_mask`.
        2. Build augmented sigma points via
           :meth:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF.make_pts_and_wts`.
        3. Propagate each sigma point through nonlinear dynamics using
           :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.noiseless_rk4`.
        4. Predict measurements for each propagated sigma point using
           :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.sensor_readings`
           with :class:`~ADCS.satellite_hardware.errors.ErrorMode` selecting bias inclusion.
        5. Compute predicted mean/covariances using unscented weights.
        6. Solve for Kalman gain, apply innovation update, and map the reduced attitude
           error back to a full quaternion state.

        Orbital interpolation (CG5)
        ---------------------------
        This implementation computes intermediate orbital states between
        :attr:`prev_os` and the current :class:`~ADCS.orbits.orbital_state.Orbital_State`
        using coefficients from :class:`~ADCS.orbits.universal_constants.CG5`:

        .. math::

            \mathbf{o}_j = \operatorname{average}(\mathbf{o}_{prev}, \mathbf{o}, c_j),\qquad j=0,\ldots,4,

        via :meth:`~ADCS.orbits.orbital_state.Orbital_State.average`, where
        :math:`c_j` are the CG5 coefficients.

        Unscented prediction and update
        -------------------------------
        Let propagated reduced sigma points be :math:`\chi_i^-` and predicted measurement
        sigma points be :math:`\gamma_i^-`. Predicted means:

        .. math::

            \bar{\mathbf{x}}^- = \sum_{i=0}^{2L} W_i^{(m)} \chi_i^-,\qquad
            \bar{\mathbf{y}}^- = \sum_{i=0}^{2L} W_i^{(m)} \gamma_i^-.

        Deviations:

        .. math::

            \mathbf{X}_i = \chi_i^- - \bar{\mathbf{x}}^-,\qquad
            \mathbf{Y}_i = \gamma_i^- - \bar{\mathbf{y}}^-.

        Covariances:

        .. math::

            P^- = \sum_{i=0}^{2L} W_i^{(c)} \mathbf{X}_i\mathbf{X}_i^\top + Q,

        .. math::

            P_{yy} = \sum_{i=0}^{2L} W_i^{(c)} \mathbf{Y}_i\mathbf{Y}_i^\top + R,

        .. math::

            P_{y x} = \sum_{i=0}^{2L} W_i^{(c)} \mathbf{Y}_i\mathbf{X}_i^\top.

        Gain and update (note the implementation’s array orientation):

        .. math::

            K = P_{yy}^{-1} P_{y x},

        and with innovation :math:`\boldsymbol{\nu}=\mathbf{y}-\bar{\mathbf{y}}^-`:

        .. math::

            \bar{\mathbf{x}}^+ = \bar{\mathbf{x}}^- + \boldsymbol{\nu}^\top K,

        .. math::

            P^+ = P^- - K^\top P_{yy} K.

        Quaternion reconstruction
        -------------------------
        In reduced attitude-error mode (``quat_as_vec=False``), the updated reduced
        vector contains an attitude error :math:`\delta\boldsymbol{\theta}` which is
        mapped to a quaternion increment and composed with the propagated reference
        quaternion:

        .. math::

            \delta\mathbf{q} = \operatorname{vec3\_to\_quat}(\delta\boldsymbol{\theta}),\qquad
            \mathbf{q}^+ = \mathbf{q}_{ref} \otimes \delta\mathbf{q},

        using :func:`~ADCS.helpers.math_helpers.vec3_to_quat` and
        :func:`~ADCS.helpers.math_helpers.quat_mult`.

        If ``quat_as_vec=True``, the quaternion is renormalized with
        :func:`~ADCS.helpers.math_helpers.normalize` and the covariance is
        transformed by the normalization Jacobian from
        :func:`~ADCS.helpers.math_helpers.state_norm_jac`:

        .. math::

            P^+ \leftarrow J^\top P^+ J.

        :param u: Control input vector.
        :type u: numpy.ndarray
        :param sensors: Stacked array of actual sensor measurements.
        :type sensors: numpy.ndarray
        :param os: Current orbital environment state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :raises numpy.linalg.LinAlgError: If the measurement covariance matrix is singular during gain computation.
        :return: Updated estimate container with updated full state and reduced covariance.
        :rtype: :class:`~ADCS.state.EstimatorState`

        """
        u = np.asarray(u, dtype=float).copy()
        os = os.copy()
        state0 = self.x_hat.as_estimator_array()

        # Middle orbital states (CG5 coefficients)
        mid_os = [self.prev_os.average(os, CG5.c[j]) for j in range(5)]

        # One-step propagation of the nominal dynamics
        dyn_state0 = self.est_sat.noiseless_rk4(
            x=self.x_hat,
            u=u,
            dt=self.dt,
            orbital_state0=self.prev_os,
            orbital_state1=os,
            mid_orbital_state=mid_os,
            quat_as_vec=False,
        )

        # Determine which attitude sensors are active based on ACTUAL measurements
        # We check the actual sensor readings for NaN, not the estimated predictions,
        # because the true satellite state may differ from our estimate.
        which_sensors = [True] * len(self.est_sat.attitude_sensors + self.est_sat.rw_actuators)
        sensor_idx = 0
        for j, sensor in enumerate(self.est_sat.attitude_sensors):
            output_len = sensor.output_length
            # Check if the actual measurement contains NaN
            sensor_reading = sensors[sensor_idx:sensor_idx + output_len]
            if np.isnan(sensor_reading).any():
                which_sensors[j] = False
            sensor_idx += output_len

        # Expand sensor mask to output mask (handles sensors with output_length > 1)
        which_outputs = self._expand_sensor_mask(which_sensors)

        # Total attitude measurement dimension
        sens_vec_len = sum(
            self.est_sat.sensors[j].output_length
            for j in range(len(self.est_sat.sensors))
            if which_sensors[j]
        )
        sens_vec_len += len(self.est_sat.rw_actuators)

        # Sigma points of augmented state
        L, pts, wts_m, wts_c, sig0 = self.make_pts_and_wts(state0, which_sensors)
        num_sigma = 2 * L + 1

        sigma_state_len = state0.size - 1 + self.quat_as_vec
        post_pts = np.empty((num_sigma, sigma_state_len), dtype=float)
        post_sens = np.empty((num_sigma, sens_vec_len), dtype=float)

        # Local aliases to avoid attribute lookups in the loop
        est_sat_template = self.est_sat
        state_len = est_sat_template.state_len

        satj = copy.deepcopy(est_sat_template)

        post_quat = None

        # Propagate each sigma point (this part is hard to vectorize because
        # it calls user dynamics; kept as a tight Python loop).
        for j in range(num_sigma):
            full_pre_array_j, sens_noise_j, control_noise_j, int_noise_extra_j = pts[j]
            full_pre_statej = self._from_augmented_array(full_pre_array_j)

            self.sat_match(satj, full_pre_statej)

            post_dyn_state_j = satj.noiseless_rk4(
                x=full_pre_statej,
                u=u + control_noise_j,
                dt=self.dt,
                orbital_state0=self.prev_os,
                orbital_state1=os,
                mid_orbital_state=mid_os,
                quat_as_vec=False,
            )

            if j == 0:
                # j=0 has zero integration noise, so this reference quaternion
                # is clean (same as original implementation).
                post_quat = post_dyn_state_j.q.copy()

            post_statej, post_full_statej = self.new_post_state(
                full_pre_array_j[state_len:],
                post_dyn_state_j.as_array(),
                int_noise_extra_j,
                post_quat,
            )
            post_pts[j, :] = post_statej.copy()

            post_full_state_obj = self._from_augmented_array(post_full_statej)
            self.sat_match(satj, post_full_state_obj)
            dmode = ErrorMode(add_bias=True, add_noise=False, update_bias=False, update_noise=False)
            sensj = satj.sensor_readings(x=post_full_state_obj,os=os, dmode=dmode)
            post_sens[j, :] = sensj[which_outputs] + sens_noise_j

        # Predicted state in the covariance coordinate convention.
        state1 = np.dot(wts_m, post_pts)
        if not self.quat_as_vec:
            dquat1 = vec3_to_quat(state1[3:6], self.vec_mode)
            quat1 = quat_mult(post_quat, dquat1)
            pred_dyn_state = np.concatenate(
                (
                    state1[0:3],
                    quat1,
                    state1[6 : state_len - 1],
                    state1[state_len - 1 :],
                )
            )
        else:
            state1[3:7] = normalize(state1[3:7])
            pred_dyn_state = state1.copy()

        # Predicted measurement
        sens1 = wts_m @ post_sens

        # Differences from mean
        pts_diff = post_pts - state1
        sens_diff = post_sens - sens1

        # Covariance of state: sum w_i * dx_i dx_i^T = X^T W X
        cov1 = sum([wts_c[j]*np.outer(pts_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)])
        cov1 += self.x_hat.int_cov

        # Measurement covariance
        covyy = sum([wts_c[j]*np.outer(sens_diff[j,:],sens_diff[j,:]) for j in range(2*L+1)],0*np.eye(sens_vec_len))
        covyy += self.est_sat.sensor_cov(which_sensors=which_sensors)

        # Cross covariance between measurements and state
        covyx = sum([wts_c[j]*np.outer(sens_diff[j,:],pts_diff[j,:]) for j in range(2*L+1)],np.zeros((sens_vec_len,sigma_state_len)))

        # Kalman gain: covyy * Kk = covyx
        try:
            Kk = scipy.linalg.solve(covyy, covyx)
        except Exception as e:
            raise np.linalg.LinAlgError("Matrix is singular. (probably)") from e

        # Innovation (same as original: residual @ Kk, where Kk is m x n)
        y_meas = sensors[which_outputs]
        innov = y_meas - sens1

        # Updated reduced error state
        state2 = state1 + innov @ Kk
        cov2 = cov1 - Kk.T @ covyy @ Kk
        # Enforce symmetry
        cov2 = 0.5 * (cov2 + cov2.T)

        # Map reduced error state back to full state representation
        if not self.quat_as_vec:
            dvec3 = state2[3:6]
            dquat = vec3_to_quat(dvec3, self.vec_mode)
            quat = quat_mult(post_quat, dquat)
            state2 = np.concatenate(
                (
                    state2[0:3],
                    quat,
                    state2[6 : state_len - 1],
                    state2[state_len - 1 :],
                )
            )
        else:
            state20 = state2.copy()
            state2[3:7] = normalize(state2[3:7])
            norm_jac = state_norm_jac(state20)
            cov2 = norm_jac.T @ cov2 @ norm_jac

        # Update satellite estimate for the new full state
        result = self._from_augmented_array(
            state2,
            cov=cov2,
            int_cov=self.x_hat.int_cov,
        )
        self.sat_match(satj, result)

        return result
