__all__ = ["SRUAKF"]

import numpy as np
import scipy.linalg
import copy
from typing import List, Tuple, Optional, Sequence
import time

# Rank-1 Cholesky update/downdate. Formerly the external `choldate` package,
# which has no PyPI release and had to be installed from git -- which made the
# whole package impossible to `pip install` cleanly. Now in-tree, matching
# choldate to ~1e-15 but raising NaN on a non-positive-definite downdate where
# choldate silently returned a wrong factor (see ADCS/helpers/cholesky_update).
from ADCS.helpers.cholesky_update import cholupdate, choldowndate

from ADCS.state import EstimatorState
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import SunSensor, SunPair
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State, Ephemeris
from ADCS.orbits.universal_constants import CG5
from ADCS.helpers.math_helpers import (
    vec3_to_quat,
    quat_mult,
    normalize,
    state_norm_jac,
)

# Import the base UKF
from ADCS.estimators.old_attitude_estimators.attitude_UAKF import UAKF

class SRUAKF(UAKF):
    r"""
    Square Root Unscented Kalman Filter (SR-UKF) using ``choldate`` rank-1 updates.

    This class implements a numerically stable Square Root UKF (SR-UKF) for
    spacecraft attitude estimation with an augmented state and reduced attitude
    error representation inherited from :class:`~ADCS.estimators.old_attitude_estimators.attitude_UAKF.UAKF`.

    Instead of propagating the covariance matrix :math:`P`, this filter
    propagates an **upper-triangular** Cholesky factor :math:`S` such that

    .. math::

        P = S^\top S,

    which improves numerical stability and helps keep :math:`P` symmetric
    positive semi-definite under finite precision arithmetic.

    This implementation uses the external C-extension ``choldate`` for fast
    rank-1 Cholesky updates and downdates, which is particularly important
    for the SR-UKF because the 0th sigma point can carry a negative covariance
    weight :math:`W_0^{(c)} < 0`, requiring a Cholesky **downdate**.

    State and reduced error representation
    --------------------------------------
    Let the (possibly augmented) estimator state be :math:`\mathbf{x}`. In the
    common reduced-attitude-error formulation, the quaternion is not directly
    included as 4 DOF in the covariance; instead, a 3-parameter error vector
    :math:`\delta\boldsymbol{\theta}\in\mathbb{R}^3` is used, so the reduced
    covariance dimension is :math:`L_x = N-1` (unless ``quat_as_vec=True``).

    Process noise is represented by a covariance :math:`Q` with square root
    :math:`S_Q`:

    .. math::

        Q = S_Q^\top S_Q.

    See also:
    - :class:`~ADCS.estimators.old_attitude_estimators.attitude_UAKF.UAKF`
    - :class:`~ADCS.state.EstimatorState`
    - :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

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
        ephem: Optional[Ephemeris] = None,
    ) -> None:
        r"""
        Initialize the SR-UKF and compute initial square-root factors.

        This constructor delegates base initialization to
        :class:`~ADCS.estimators.old_attitude_estimators.attitude_UAKF.UAKF` and then
        computes:

        - the upper-triangular Cholesky factor :math:`S` of the current state
          covariance :math:`P` stored in :attr:`x_hat.cov`,
        - the upper-triangular Cholesky factor :math:`S_Q` of the process noise
          covariance :math:`Q` stored in :attr:`x_hat.int_cov`.

        Cholesky factorization attempts
        --------------------------------
        For each covariance matrix :math:`M\in\{P, Q\}`, the algorithm attempts
        a Cholesky factorization first:

        .. math::

            M = L L^\top,\quad S = L^\top.

        If Cholesky fails (e.g., :math:`M` not strictly positive definite), a
        fallback eigen-decomposition is used to form a symmetric square root:

        .. math::

            M = V \Lambda V^\top,\quad
            \sqrt{M} = V \sqrt{|\Lambda|}\, V^\top,\quad
            S = (\sqrt{M})^\top,

        where :math:`\sqrt{|\Lambda|}` is formed elementwise on the diagonal.

        :param est_sat: Estimated satellite model defining state structure and dynamics.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param J2000: Initial time in seconds since J2000.
        :type J2000: float
        :param x_hat: Initial full augmented state estimate.
        :type x_hat: ADCS.state.EstimatorState
        :param P_hat: Initial reduced error-state covariance matrix.
        :type P_hat: numpy.ndarray
        :param Q_hat: Initial reduced process-noise covariance matrix.
        :type Q_hat: numpy.ndarray
        :param dt: Propagation time step :math:`\Delta t` in seconds.
        :type dt: float
        :param cross_term: Whether cross-covariances between bias/parameter blocks are preserved.
        :type cross_term: bool
        :param quat_as_vec: If ``True``, treat quaternion as a 4-vector in the covariance.
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
            ephem=ephem,
        )

        try:
            # numpy cholesky returns Lower, so we transpose to get Upper
            self.S = np.linalg.cholesky(self.x_hat.cov).T
        except np.linalg.LinAlgError:
            # Fallback to Eigendecomposition if P is not strictly positive definite
            w, v = np.linalg.eig(self.x_hat.cov)
            srw = np.diag(np.sqrt(np.abs(np.real(w))))
            v = np.real(v)
            # Construct Upper Triangular S s.t. S.T @ S = P
            # P = V D V.T = (V D^0.5) (V D^0.5).T
            # We want S such that S = (V D^0.5).T
            self.S = (v @ srw @ v.T).T

        # Initialize Process Noise Square Root (Upper Triangular)
        try:
            self.S_Q = np.linalg.cholesky(self.x_hat.int_cov).T
        except np.linalg.LinAlgError:
            w, v = np.linalg.eig(self.x_hat.int_cov)
            srw = np.diag(np.sqrt(np.abs(np.real(w))))
            v = np.real(v)
            self.S_Q = (v @ srw @ v.T).T

    def weighted_cholupdate(self, mat: np.ndarray, vec: np.ndarray, wt: float) -> np.ndarray:
        r"""
        Perform a weighted rank-1 Cholesky update/downdate using ``choldate``.

        This method updates an **upper-triangular** Cholesky factor :math:`S`
        (named ``mat``) to reflect adding or subtracting a weighted outer product
        of a vector :math:`\mathbf{v}`:

        - For :math:`wt \ge 0` (update):

          .. math::

              S_{\text{new}}^\top S_{\text{new}}
              =
              S^\top S + wt\, \mathbf{v}\mathbf{v}^\top.

        - For :math:`wt < 0` (downdate):

          .. math::

              S_{\text{new}}^\top S_{\text{new}}
              =
              S^\top S - |wt|\, \mathbf{v}\mathbf{v}^\top.

        Implementation detail
        ---------------------
        The ``choldate`` routines operate in-place on the Cholesky factor and
        accept the scaled vector :math:`\sqrt{|wt|}\,\mathbf{v}`.

        This wrapper provides:

        - sign logic: :func:`choldate.cholupdate` if :math:`wt\ge 0`,
          :func:`choldate.choldowndate` if :math:`wt < 0`,
        - handling of matrix inputs ``vec`` (multiple vectors) by applying
          sequential rank-1 operations.

        Matrix input semantics
        ----------------------
        If ``vec`` is a 2D array with multiple vectors, the method applies

        .. math::

            S \leftarrow \operatorname{cholop}(S, \mathbf{v}_1, wt),\;
            S \leftarrow \operatorname{cholop}(S, \mathbf{v}_2, wt),\;\ldots

        where :math:`\operatorname{cholop}` is an update or downdate depending on
        the sign of :math:`wt`.

        :param mat: Upper-triangular Cholesky factor :math:`S` to be modified.
        :type mat: numpy.ndarray
        :param vec: Update vector :math:`\mathbf{v}` (shape ``(n,)``) or a matrix
            of vectors (shape ``(k,n)`` or ``(n,k)``) to be applied sequentially.
        :type vec: numpy.ndarray
        :param wt: Scalar weight controlling update (nonnegative) or downdate (negative).
        :type wt: float
        :raises ValueError: If ``vec`` has more than 2 dimensions.
        :raises numpy.linalg.LinAlgError: If NaN/Inf appears in the updated factor.
        :return: Updated upper-triangular Cholesky factor :math:`S_{\text{new}}`.
        :rtype: numpy.ndarray

        """
        mat = mat.copy()
        vec = vec.copy()
        s = np.shape(vec)
        
        # Validation
        if len(s) > 2:
            raise ValueError('weighted_cholupdate: Can only handle vectors and matrices')
        
        # Handle Matrix Case (Update using multiple vectors)
        # Check if 2D array and not a single row/col vector
        if len(s) == 2 and not (s[0] == 1 or s[1] == 1):
            # We expect vec to be shape (N_vectors, N_state) matching mat columns
            # If dimensions look swapped, transpose
            rr = s[0]
            if s[1] != mat.shape[0]: 
                # This assumes we wanted to update with columns
                vec = vec.T.copy()
                rr = s[1]
            
            # Recursive update for each row in the vector matrix
            for j in range(rr):
                mat = self.weighted_cholupdate(mat, vec[j, :], wt)
                if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
                    raise np.linalg.LinAlgError('NAN encountered in weighted_cholupdate')
        
        # Handle Vector Case
        else:
            vec = np.ravel(vec)
            if wt >= 0:
                cholupdate(mat, (wt**0.5) * vec)
            else:
                # Note: choldate is in-place
                choldowndate(mat, ((-wt)**0.5) * vec)
                
        return mat

    def make_pts_and_wts(self, pt0: np.ndarray, which_sensors: Sequence[bool]):
        r"""
        Generate augmented sigma points using the stored square root covariance.

        This method generates sigma points for a UKF-style estimator, while
        exploiting the persistently tracked square-root factor :math:`S` of the
        reduced covariance for efficiency.

        It produces the standard UKF outputs:

        - augmented dimension :math:`L`,
        - sigma point container list ``pts`` of length :math:`2L+1`,
        - mean weights :math:`W_i^{(m)}`,
        - covariance weights :math:`W_i^{(c)}`,
        - state-only sigma points array ``sig0`` aligned with ``pts``.

        Unscented transform weights
        ---------------------------
        Let :math:`L` be the augmented dimension. With standard UKF parameters
        :math:`\alpha`, :math:`\beta`, :math:`\kappa` (stored in the base class),
        define

        .. math::

            \lambda = \alpha^2 (L + \kappa) - L,\qquad
            \gamma = \sqrt{L + \lambda}.

        The weights are

        .. math::

            W_0^{(m)} = \frac{\lambda}{L+\lambda},\qquad
            W_0^{(c)} = \frac{\lambda}{L+\lambda} + (1-\alpha^2+\beta),

        .. math::

            W_i^{(m)} = W_i^{(c)} = \frac{1}{2(L+\lambda)},\quad i=1,\ldots,2L.

        Sigma point construction with square root factor
        ------------------------------------------------
        For the (reduced) state block of size :math:`L_x`, the covariance is

        .. math::

            P_x = S^\top S,

        where :math:`S` is **upper-triangular**. If :math:`L_x = \dim(\mathbf{x})`
        in the reduced space, sigma-point offsets are formed from the rows of
        :math:`S` scaled by :math:`\gamma`:

        .. math::

            \Delta X = \gamma S.

        These offsets are applied to the mean state :math:`\bar{\mathbf{x}}`
        using the estimator-specific state addition (which must respect quaternion
        error composition), via the inherited method
        :meth:`~ADCS.estimators.old_attitude_estimators.attitude_UAKF.UAKF.add_to_state`.

        Augmentation blocks
        -------------------
        The method supports an augmented space including:

        - state error block (from :math:`S`),
        - control/process noise block (from :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.control_cov`),
        - sensor noise block (shape tracked but treated as zero contribution in this architecture),
        - integration noise block (shape tracked but treated as zero contribution here).

        For the control/process noise block with covariance :math:`Q_u`, sigma
        offsets are formed from a Cholesky factor :math:`L_u` such that
        :math:`Q_u = L_u L_u^\top`.

        Output alignment
        ----------------
        The returned ``pts`` list stores each sigma point as a 4-list:

        .. math::

            [\text{full\_state},\; \text{sens\_noise},\; \text{control\_noise},\; \text{int\_noise}].

        The array ``sig0`` is a stacked view of the state component only, aligned
        in order with ``pts``; sigma points whose perturbation is purely in noise
        blocks are padded with the mean state.

        :param pt0: Current mean full state vector (augmented state representation).
        :type pt0: numpy.ndarray
        :param which_sensors: Boolean mask indicating which sensors are active.
        :type which_sensors: list[bool]
        :return: Tuple ``(L, pts, wts_m, wts_c, sig0)`` where:

            - ``L`` is the augmented dimension,
            - ``pts`` is the sigma point list,
            - ``wts_m`` are mean weights,
            - ``wts_c`` are covariance weights,
            - ``sig0`` is the stacked state-only sigma point array.

        :rtype: tuple[int, list, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        # 1. Setup Covariances & Dimensions
        # ---------------------------------
        # State Covariance (Error State)
        # Use self.S directly later, but we need the dimension here.
        L_x = self.S.shape[0]

        # Control/Process Noise Covariance
        control_cov = self.est_sat.control_cov()
        
        # Sensor & Integration Covariances (Zeroed out in this architecture)
        # We maintain their shapes for the list structure.
        sens_cov = self.est_sat.sensor_cov(which_sensors=which_sensors) * 0.0
        int_cov = self.x_hat.int_cov * 0.0

        include_cov = self.determine_covariances_to_use(
            self.x_hat.cov,
            sens_cov,
            control_cov,
            int_cov,
        )
        covs = [self.x_hat.cov, sens_cov, control_cov, int_cov]

        L_q = control_cov.shape[0] if include_cov[2] and control_cov.size > 0 else 0

        # Total Augmented Dimension must match the blocks that actually generate points.
        L = int(sum(include_cov[j] * covs[j].shape[0] for j in range(4)))

        # 2. Weights (Standard UKF)
        # -------------------------
        self.lam = self.al ** 2.0 * (self.kap + L) - L
        gamma = np.sqrt(L + self.lam)
        
        denom = L + self.lam
        w0_m = self.lam / denom
        w0_c = self.lam / denom + (1.0 - self.al ** 2.0 + self.bet)
        wi = 0.5 / denom
        
        num_sigma = 2 * L + 1
        wts_m = np.full(num_sigma, wi)
        wts_c = np.full(num_sigma, wi)
        wts_m[0] = w0_m
        wts_c[0] = w0_c
        
        self.wts_m = wts_m
        self.wts_c = wts_c

        # 3. Construct "Zeros" for the list structure
        # -------------------------------------------
        # Ensure we return valid numpy arrays even if empty
        dtype = pt0.dtype
        zeros_state = pt0 # Not used as zero, but as placeholder
        zeros_sens = np.zeros(sens_cov.shape[0], dtype=dtype) if sens_cov.size > 0 else np.zeros(0, dtype=dtype)
        zeros_ctrl = np.zeros(L_q, dtype=dtype) if L_q > 0 else np.zeros(0, dtype=dtype)
        zeros_int  = np.zeros(int_cov.shape[0], dtype=dtype) if int_cov.size > 0 else np.zeros(0, dtype=dtype)

        # The mean point list [state, sens, ctrl, int]
        # Note: pt0 is the mean state
        zeros = [pt0, zeros_sens, zeros_ctrl, zeros_int]
        pts = [zeros]

        # 4. Generate Sigma Points Block by Block
        # ---------------------------------------
        
        # --- BLOCK 1: STATE (Uses self.S directly) ---
        # S is Upper Triangular, so P = S.T @ S. 
        # The Cholesky L (Lower) is S.T.
        # Sigma offsets are columns of L, which are rows of S.
        scaled_S = gamma * self.S
        
        # Offsets: [+S_rows, -S_rows]
        state_offsets = np.vstack((scaled_S, -scaled_S))
        
        # Apply offsets to state (handling quaternions)
        sig_states = self.add_to_state(pt0, state_offsets)
        
        # Append to pts: [modified_state, 0, 0, 0]
        for k in sig_states:
            pts.append([k, zeros_sens, zeros_ctrl, zeros_int])

        # --- BLOCK 2: SENSORS (Skipped - assumed zero) ---
        # If L_r > 0, we would compute cholesky(sens_cov) here.

        # --- BLOCK 3: CONTROL / PROCESS NOISE ---
        if include_cov[2] and L_q > 0:
            try:
                root_q = np.linalg.cholesky(control_cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(control_cov)
                root_q = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))

            scaled_root_q = gamma * root_q
            q_offsets = np.hstack((scaled_root_q, -scaled_root_q)).T

            for k in q_offsets:
                pts.append([pt0, zeros_sens, k, zeros_int])

        # --- BLOCK 4: INT NOISE (Skipped - assumed zero) ---

        # 5. Construct sig0 (State components only, padded)
        # -------------------------------------------------
        # sig0 is a (2L+1, state_dim) array used for fast vectorized operations
        # on the state part of the sigma points.
        # It must align with 'pts'. 
        # Structure: [Mean, State_Pts, Mean_Repeated_for_Noise_Pts]
        
        sig0_list = [pt0]
        sig0_list.extend(sig_states) # The state-perturbed points
        
        # Determine how many remaining points are just the mean state
        # (These are the points perturbed by Control/Sensor noise)
        current_count = len(sig0_list)
        pad_count = num_sigma - current_count
        
        if pad_count > 0:
            # Efficiently repeat the mean
            sig0_list.extend([pt0] * pad_count)
            
        sig0 = np.vstack(sig0_list)

        return L, pts, wts_m, wts_c, sig0

    def update_core(
        self,
        u: np.ndarray,
        sensors: np.ndarray,
        os: Orbital_State,
    ) -> EstimatorState:
        r"""
        Perform one SR-UKF predict/update cycle using QR factorization and Cholesky downdates.

        This method executes a full SR-UKF cycle in square-root form:

        1. **Time update (prediction)**: propagate sigma points through dynamics and
           compute the predicted mean and predicted square-root covariance :math:`S^-`.
        2. **Measurement update**: predict measurements, compute innovation covariance
           square root :math:`S_{yy}`, compute the Kalman gain, update the mean.
        3. **Square-root covariance update**: apply sequential Cholesky downdates using
           :func:`choldate.choldowndate` via :meth:`weighted_cholupdate`.

        The method returns an :class:`~ADCS.state.EstimatorState`
        containing the updated full state and a reconstructed covariance :math:`P^+`.

        Notation
        --------
        Let sigma points be :math:`\chi_i` with weights :math:`W_i^{(m)}`, :math:`W_i^{(c)}`.
        Let the propagated sigma points be :math:`\chi_i^-` and the predicted mean be
        :math:`\bar{\mathbf{x}}^-`.

        Let measurement sigma points be :math:`\gamma_i^- = h(\chi_i^-)` with predicted
        mean :math:`\bar{\mathbf{y}}^-`.

        Sigma point propagation
        -----------------------
        Dynamics propagation is performed using the satellite model RK4 integrator:

        - :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_rk4`

        for each sigma point, with control perturbations applied where relevant.

        Predicted mean
        --------------
        The predicted reduced mean is formed by the unscented transform:

        .. math::

            \bar{\mathbf{x}}^- = \sum_{i=0}^{2L} W_i^{(m)} \chi_i^-.

        Quaternion handling
        -------------------
        In reduced attitude-error mode (``quat_as_vec=False``), sigma points encode
        a 3-vector attitude error :math:`\delta\boldsymbol{\theta}` (stored in the
        reduced state) which is mapped to a quaternion increment via
        :func:`~ADCS.helpers.math_helpers.vec3_to_quat` and composed with the
        propagated reference quaternion using :func:`~ADCS.helpers.math_helpers.quat_mult`:

        .. math::

            \delta\mathbf{q} = \operatorname{vec3\_to\_quat}(\delta\boldsymbol{\theta}),\qquad
            \mathbf{q} = \mathbf{q}_{ref} \otimes \delta\mathbf{q}.

        Predicted square-root covariance via QR
        ---------------------------------------
        Let deviations be

        .. math::

            \mathbf{X}_i = \chi_i^- - \bar{\mathbf{x}}^-.

        Form the weighted deviation matrix excluding the 0th point:

        .. math::

            A = \begin{bmatrix}
                \sqrt{W_1^{(c)}}\,\mathbf{X}_1 & \cdots & \sqrt{W_{2L}^{(c)}}\,\mathbf{X}_{2L}
            \end{bmatrix}^\top.

        Augment with the process-noise square root (transposed to align shapes)
        and compute an upper-triangular factor via QR:

        .. math::

            \begin{bmatrix} A & S_Q^\top \end{bmatrix}
            = QR,\qquad S^- \leftarrow R.

        Because :math:`W_0^{(c)}` may be negative, the 0th point is incorporated
        via a rank-1 update/downdate:

        .. math::

            S^- \leftarrow \operatorname{cholop}\!\left(S^-, \mathbf{X}_0, W_0^{(c)}\right),

        where :math:`\operatorname{cholop}` is implemented by
        :meth:`~ADCS.estimators.old_attitude_estimators.attitude_SRUAKF.SRUAKF.weighted_cholupdate`.

        Measurement square-root covariance
        ----------------------------------
        Similarly, with measurement deviations

        .. math::

            \mathbf{Y}_i = \gamma_i^- - \bar{\mathbf{y}}^-,

        and sensor noise covariance :math:`R = S_R^\top S_R`, compute

        .. math::

            \begin{bmatrix}
                \sqrt{W_1^{(c)}}\,\mathbf{Y}_1 & \cdots & \sqrt{W_{2L}^{(c)}}\,\mathbf{Y}_{2L} & S_R^\top
            \end{bmatrix}
            = QR,\qquad S_{yy} \leftarrow R,

        then incorporate :math:`\mathbf{Y}_0` with the weighted rank-1 operation.

        Cross-covariance and Kalman gain
        --------------------------------
        The cross-covariance is computed as

        .. math::

            P_{y x} = \sum_{i=0}^{2L} W_i^{(c)}\, \mathbf{Y}_i\,\mathbf{X}_i^\top.

        The Kalman gain is obtained by solving

        .. math::

            K = P_{x y}\, P_{y y}^{-1},

        without explicitly forming :math:`P_{yy}` by using triangular solves with
        :math:`S_{yy}` (upper-triangular):

        .. math::

            P_{yy} = S_{yy}^\top S_{yy},

        and for each right-hand side using forward/backward substitution. In code,
        this corresponds to calls equivalent to
        :func:`scipy.linalg.solve_triangular`.

        Innovation and mean update
        --------------------------
        With measurement :math:`\mathbf{y}`, innovation

        .. math::

            \boldsymbol{\nu} = \mathbf{y} - \bar{\mathbf{y}}^-,

        and mean update

        .. math::

            \bar{\mathbf{x}}^+ = \bar{\mathbf{x}}^- + K\,\boldsymbol{\nu}.

        Square-root covariance update (downdate form)
        ---------------------------------------------
        The SR-UKF covariance update can be expressed in square-root downdate form.
        Let

        .. math::

            U = S_{yy}\,K^\top.

        Then

        .. math::

            P^+ = P^- - K P_{yy} K^\top
                = S^{-\top} S^- - U^\top U.

        In square-root form, this is implemented by sequential Cholesky downdates:

        .. math::

            S^+ \leftarrow \operatorname{choldowndate}(S^-, U).

        This method performs that operation via
        :meth:`~ADCS.estimators.old_attitude_estimators.attitude_SRUAKF.SRUAKF.weighted_cholupdate`
        with a negative weight.

        Reconstruction and compatibility
        --------------------------------
        For compatibility with other estimator infrastructure, the method
        reconstructs a covariance matrix

        .. math::

            P^+ = S^{+\top} S^+,

        symmetrizes it, and returns it in the resulting
        :class:`~ADCS.state.EstimatorState`.

        If ``quat_as_vec=True``, the quaternion is renormalized using
        :func:`~ADCS.helpers.math_helpers.normalize` and covariance is transformed
        with the normalization Jacobian from :func:`~ADCS.helpers.math_helpers.state_norm_jac`.

        Sensor activation logic
        -----------------------
        Active attitude sensors are inferred from the *actual* measurement vector
        by checking each sensor output slice for NaNs. The sensor-output mask is
        expanded using the inherited helper method
        :meth:`~ADCS.estimators.old_attitude_estimators.attitude_UAKF.UAKF._expand_sensor_mask`.

        :param u: Control input vector.
        :type u: numpy.ndarray
        :param sensors: Stacked array of actual sensor measurements.
        :type sensors: numpy.ndarray
        :param os: Current orbital environment state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :raises numpy.linalg.LinAlgError: If the innovation covariance square root is singular
            during triangular solves.
        :return: Updated estimate container with fields:

            - full augmented state,
            - ``cov``: reconstructed reduced covariance :math:`P^+`,
            - ``int_cov``: unchanged (inherited from internal state).

        :rtype: :class:`~ADCS.state.EstimatorState`

        """
        u = np.asarray(u, dtype=float).copy()
        os = os.copy()
        state0 = self.x_hat.as_estimator_array()

        # --- 1. Determine Active Sensors (Eclipse Check) ---
        mid_os = [self.prev_os.average(os, CG5.c[j]) for j in range(5)]
        
        # Nominal propagation
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

        # --- 2. Sigma Point Generation ---
        sens_vec_len = sum(
            self.est_sat.sensors[j].output_length
            for j in range(len(self.est_sat.sensors))
            if which_sensors[j]
        )
        sens_vec_len += len(self.est_sat.rw_actuators)

        L, pts, wts_m, wts_c, sig0 = self.make_pts_and_wts(state0, which_sensors)
        num_sigma = 2*L+1
        
        sigma_state_len = state0.size - 1 + self.quat_as_vec
        post_pts = np.empty((num_sigma, sigma_state_len), dtype=float)
        post_sens = np.empty((num_sigma, sens_vec_len), dtype=float)

        est_sat_template = self.est_sat
        state_len = est_sat_template.state_len
        satj = copy.deepcopy(est_sat_template)
        post_quat = None

        # --- 3. Propagation Loop ---
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

        # --- 4. Time Update (QR Method) ---
        state1 = wts_m @ post_pts
        if not self.quat_as_vec:
            dquat1 = vec3_to_quat(state1[3:6], self.vec_mode)
            quat1 = quat_mult(post_quat, dquat1)
        else:
            state1[3:7] = normalize(state1[3:7])
        
        # State deviations
        pts_diff = post_pts - state1

        weighted_diffs = pts_diff[1:, :].T * np.sqrt(wts_c[1:])
        
        to_qr = np.hstack([pts_diff[1:, :].T * np.sqrt(wts_c[1:]), self.S_Q.T])
        # Note: using S_Q.T implies S_Q was Upper, so S_Q.T is Lower. 
        
        srcov1 = np.linalg.qr(to_qr.T, mode='r')
        
        # Handle index 0 (negative weight)
        srcov1 = self.weighted_cholupdate(srcov1, pts_diff[0, :], wts_c[0])

        # --- 5. Measurement Update (QR Method) ---
        sens1 = wts_m @ post_sens
        sens_diff = post_sens - sens1
        
        # Sensor Noise S_R (Upper Triangular)
        R_cov = self.est_sat.sensor_cov(which_sensors=which_sensors)
        try:
            S_R = np.linalg.cholesky(R_cov).T
        except:
            w,v = np.linalg.eig(R_cov)
            S_R = (np.real(v) @ np.diag(np.sqrt(np.abs(np.real(w)))) @ np.real(v).T).T

        # Measurement QR
        to_qr_sens = np.hstack([sens_diff[1:, :].T * np.sqrt(wts_c[1:]), S_R.T])
        srcov_sens = np.linalg.qr(to_qr_sens.T, mode='r')
        
        # Handle index 0
        srcov_sens = self.weighted_cholupdate(srcov_sens, sens_diff[0, :], wts_c[0])

        # --- 6. Kalman Gain ---
        # Cross Covariance
        covyx = sum([wts_c[j] * np.outer(sens_diff[j, :], pts_diff[j, :]) for j in range(num_sigma)])
        
        try:
            t1 = scipy.linalg.solve_triangular(srcov_sens, covyx, lower=False, trans='T')
            Kk = scipy.linalg.solve_triangular(srcov_sens, t1, lower=False)
        except np.linalg.LinAlgError:
             raise np.linalg.LinAlgError('SRUAKF: Singular Matrix in Update')

        # --- 7. State Update ---
        # Note: Kk here is (n_meas, n_state).
        # State row vec update: x_new = x + (y - y_est) @ Kk
        y_meas = sensors[which_outputs]
        innov = y_meas - sens1
        
        state2 = state1 + innov @ Kk

        # --- 8. Covariance Update (Downdate) ---
        srcov2 = srcov1.copy()
        
        # Calculate update matrix U
        # U = srcov_sens @ Kk
        # This aligns dimensions: (n_meas, n_meas) @ (n_meas, n_state) -> (n_meas, n_state)
        U = srcov_sens @ Kk
        
        # Downdate srcov2 by the rows of U (columns of U.T)
        # Old code: srcov2 = weighted_cholupdate(srcov2, U, -1)
        srcov2 = self.weighted_cholupdate(srcov2, U, -1.0)
        
        self.S = srcov2
        
        # Reconstruct P for compatibility (P = S.T @ S)
        P_plus = self.S.T @ self.S
        P_plus = 0.5 * (P_plus + P_plus.T)

        # --- 9. Reconstruction ---
        if not self.quat_as_vec:
            dvec3 = state2[3:6]
            dquat = vec3_to_quat(dvec3, self.vec_mode)
            quat_final = quat_mult(post_quat, dquat)
            state_final = np.concatenate((
                state2[0:3],
                quat_final,
                state2[6:self.est_sat.state_len-1],
                state2[self.est_sat.state_len-1:]
            ))
        else:
            state_final = state2.copy()
            state_final[3:7] = normalize(state_final[3:7])
            norm_jac = state_norm_jac(state_final)
            P_plus = norm_jac.T @ P_plus @ norm_jac

        result = self._from_augmented_array(
            state_final,
            cov=P_plus,
            int_cov=self.x_hat.int_cov,
        )
        self.sat_match(satj, result)
        return result
