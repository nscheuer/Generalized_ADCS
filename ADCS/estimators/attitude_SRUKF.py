__all__ = ["SRUKF"]

import numpy as np
from choldate import cholupdate, choldowndate
from typing import List

from scipy.linalg import solve_triangular

from ADCS.estimators.estimator import Estimator
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import quat_to_vec3, vec3_to_quat


class SRUKF(Estimator):
    r"""
    Square-root Unscented Kalman Filter (SR-UKF) for an :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.

    This filter operates on an **augmented spacecraft state** that includes:

    .. math::

        \mathbf{x}
        =
        \begin{bmatrix}
            \boldsymbol{\omega} \\
            \mathbf{q} \\
            \mathbf{h}_{\mathrm{RW}} \\
            \mathbf{b}_{\mathrm{act}} \\
            \mathbf{b}_{\mathrm{sens}} \\
            \boldsymbol{\theta}_{\mathrm{dist}}
        \end{bmatrix}
        \in \mathbb{R}^{N},

    where

    - :math:`\boldsymbol{\omega} \in \mathbb{R}^{3}` is the body angular rate,
    - :math:`\mathbf{q} \in \mathbb{R}^{4}` is the attitude quaternion (Hamilton convention),
    - :math:`\mathbf{h}_{\mathrm{RW}} \in \mathbb{R}^{n_{\mathrm{RW}}}` are reaction wheel momenta,
    - :math:`\mathbf{b}_{\mathrm{act}} \in \mathbb{R}^{n_{\mathrm{ab}}}` are actuator bias states,
    - :math:`\mathbf{b}_{\mathrm{sens}} \in \mathbb{R}^{n_{\mathrm{sb}}}` are sensor bias states,
    - :math:`\boldsymbol{\theta}_{\mathrm{dist}} \in \mathbb{R}^{n_{\mathrm{dp}}}` are disturbance model parameters.

    The total full-state dimension is

    .. math::

        N
        =
        (7 + n_{\mathrm{RW}})
        + n_{\mathrm{ab}}
        + n_{\mathrm{sb}}
        + n_{\mathrm{dp}},

    which must match :attr:`~ADCS.satellite_hardware.satellite.satellite.Satellite.state_len`,
    :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.act_bias_len`,
    :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.att_sens_bias_len`,
    and :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.dist_param_len`.

    Internally, the SR-UKF works in a **reduced state space**
    :math:`\mathbf{y} \in \mathbb{R}^{n}` obtained by replacing the quaternion
    :math:`\mathbf{q}` with a minimal 3-parameter attitude vector
    :math:`\mathbf{v} \in \mathbb{R}^{3}`:

    .. math::

        \mathbf{y}
        =
        \begin{bmatrix}
            \boldsymbol{\omega} \\
            \mathbf{v} \\
            \mathbf{h}_{\mathrm{RW}} \\
            \mathbf{b}_{\mathrm{act}} \\
            \mathbf{b}_{\mathrm{sens}} \\
            \boldsymbol{\theta}_{\mathrm{dist}}
        \end{bmatrix}
        \in \mathbb{R}^{n}, \qquad
        n = N - 1.

    The conversion between quaternion and 3-vector attitude parameterization is
    handled by :func:`ADCS.helpers.math_helpers.quat_to_vec3` and
    :func:`ADCS.helpers.math_helpers.vec3_to_quat` using a specified
    ``vec_mode`` (e.g. modified Rodrigues parameters or other 3-DOF mapping).

    The SR-UKF is formulated in the reduced space :math:`\mathbf{y}`,
    so that the **state covariance** :math:`\mathbf{P}` is of size
    :math:`n \times n` and naturally avoids the unit-norm constraint of
    the quaternion representation.

    Key elements:

    - **Nonlinear propagation**:
      The dynamics are delegated to the associated
      :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite` via :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.noiseless_rk4`.
    - **Measurement model**:
      Sensor predictions are obtained from
      :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.noiseless_sensor_readings`.
    - **Square-root formulation**:
      Both state and process noise covariances are represented by
      their Cholesky factors :math:`\mathbf{R}` with
      :math:`\mathbf{P} \approx \mathbf{R}^\mathsf{T}\mathbf{R}`,
      which improves numerical stability.
    - **Augmented state**:
      Biases and disturbance parameters are treated as additional
      states in :math:`\mathbf{y}`, so their covariances and
      cross-covariances are propagated and updated jointly.

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        The estimated satellite model providing the nonlinear dynamics and
        measurement models as well as dimensional information:
        :attr:`~ADCS.satellite_hardware.satellite.satellite.Satellite.state_len`, :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.act_bias_len`,
        :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.att_sens_bias_len`, and :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.dist_param_len`.
    J2000 : float
        Initial epoch in J2000 seconds (passed to the base :class:`~ADCS.estimators.estimator.Estimator`
        and used to initialize the :class:`~ADCS.orbits.orbital_state.Orbital_State`).
    x_hat : :class:`numpy.ndarray`
        Initial **full** augmented state vector :math:`\mathbf{x}_0 \in \mathbb{R}^{N}`.
        Its length **must** satisfy

        .. math::

            \mathrm{len}(x\_hat) = 7 + n_{\mathrm{RW}}
            + n_{\mathrm{ab}} + n_{\mathrm{sb}} + n_{\mathrm{dp}}
            \;=\; N.
    P_hat : :class:`numpy.ndarray`
        Initial state covariance matrix :math:`\mathbf{P}_0 \in \mathbb{R}^{n\times n}`
        in the **reduced** space. Since the quaternion is 4-DOF but
        replaced by a 3-vector, the covariance dimension is one smaller
        than :math:`x\_hat`:

        .. math::

            \mathbf{P}_0 \in \mathbb{R}^{(N-1)\times(N-1)}.
    Q_hat : :class:`numpy.ndarray`
        Process noise covariance matrix :math:`\mathbf{Q} \in \mathbb{R}^{n\times n}`
        in the reduced space. Its shape must also be
        :math:`(N-1)\times(N-1)`.
    dt : float, optional
        Filter time step :math:`\Delta t` in seconds. This value is passed to
        :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.noiseless_rk4` for state propagation.
    cross_term : bool, optional
        If ``False``, certain cross-covariance blocks between actuator biases,
        sensor biases and disturbance parameters are explicitly zeroed after the
        update to enforce a block-diagonal structure in those subspaces.
        This is purely a modeling choice to decouple certain blocks.

    Attributes
    ----------
    dt : float
        Current integration time step used in the dynamics propagation.
    al : float
        UKF scaling parameter :math:`\alpha`. Controls how far sigma points
        are spread from the mean.
    kap : float
        Secondary scaling parameter :math:`\kappa`.
    bet : float
        Parameter :math:`\beta` encoding prior knowledge of the distribution.
        For Gaussian priors, :math:`\beta = 2` is optimal.
    vec_mode : int
        Mode passed to :func:`~ADCS.helpers.math_helpers.quat_to_vec3` and :func:`~ADCS.helpers.math_helpers.vec3_to_quat`
        defining the 3-DOF attitude parameterization.
    srcov : :class:`numpy.ndarray`
        Square-root state covariance :math:`\mathbf{R}_P` such that
        :math:`\mathbf{P} \approx \mathbf{R}_P^\mathsf{T}\mathbf{R}_P`.
    sric : :class:`numpy.ndarray`
        Square-root process noise covariance :math:`\mathbf{R}_Q` such that
        :math:`\mathbf{Q} \approx \mathbf{R}_Q^\mathsf{T}\mathbf{R}_Q`.

    Notes
    -----
    - All sigma-point operations are carried out in the reduced state
      space :math:`\mathbf{y}`, while the physical state fed back into
      :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite` is the full quaternion-based state
      :math:`\mathbf{x}`.
    - After each update, the function :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`
      is called, so that the internal satellite model (biases, disturbances,
      noise toggling) is consistent with the current estimate.
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
    ) -> None:
        r"""
        Initialize the square-root UKF with an augmented spacecraft state.

        This constructor simply forwards its arguments to the base
        :class:`Estimator`, caches the time step :math:`\Delta t`, sets
        Unscented Transform hyperparameters (:math:`\alpha`, :math:`\kappa`,
        :math:`\beta`), and computes initial square-root covariances
        :math:`\mathbf{R}_P` and :math:`\mathbf{R}_Q`.

        See the class docstring for a detailed description of parameter
        semantics and dimensional requirements.
        """
        
        super().__init__(
            est_sat=est_sat,
            J2000=J2000,
            x_hat=x_hat,
            P_hat=P_hat,
            Q_hat=Q_hat,
            dt=dt,
            cross_term=cross_term,
        )

        # Store step size (in case the base class doesn't)
        self.dt = dt

        # Unscented transform tuning parameters
        self.al = 1.0
        self.kap = 0.0
        self.bet = 2.0

        # Attitude 3-vector representation mode used by quat_to_vec3 / vec3_to_quat
        self.vec_mode = 6

        # Cache square-root factors of P and Q
        self._update_square_roots()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chol_from_cov(P_hat: np.ndarray) -> np.ndarray:
        r"""
        Compute an upper-triangular Cholesky factor from a covariance matrix.

        Given a symmetric positive semi-definite matrix
        :math:`\mathbf{P} \in \mathbb{R}^{k\times k}`, this function returns
        an upper-triangular matrix :math:`\mathbf{R} \in \mathbb{R}^{k\times k}`
        such that

        .. math::

            \mathbf{P} \approx \mathbf{R}^\mathsf{T} \mathbf{R}.

        If :func:`numpy.linalg.cholesky` fails (due to loss of positive
        definiteness, e.g. from numerical roundoff), a fallback based on
        eigen-decomposition is used:

        .. math::

            \mathbf{P} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\mathsf{T},
            \qquad
            \mathbf{R} = \bigl(\mathbf{V}\sqrt{\mathbf{\Lambda}}\mathbf{V}^\mathsf{T}\bigr)^\mathsf{T}.

        Parameters
        ----------
        P_hat : :class:`numpy.ndarray`
            Covariance matrix to factor, shape ``(k, k)``. For the SR-UKF,
            this is typically either :math:`\mathbf{P}` or :math:`\mathbf{Q}`
            in the reduced state space of dimension :math:`k = n = N-1`.

        Returns
        -------
        :class:`numpy.ndarray`
            Upper-triangular factor :math:`\mathbf{R}` with the same shape
            as ``P_hat`` such that :math:`\mathbf{P} \approx \mathbf{R}^\mathsf{T}\mathbf{R}`.
        """
        if P_hat.size == 0:
            return P_hat.reshape(0, 0)

        try:
            # numpy.cholesky returns lower-triangular L with P=L Lᵀ
            return np.linalg.cholesky(P_hat).T
        except np.linalg.LinAlgError:
            w, v = np.linalg.eig(P_hat)
            # Clamp tiny negative eigenvalues from numerical noise
            w = np.maximum(np.real(w), 0.0)
            srw = np.diag(np.sqrt(w))
            v = np.real(v)
            R = (v @ srw @ v.T).T
            return R

    def _update_square_roots(self) -> None:
        r"""
        Refresh the cached square-root covariances :math:`\mathbf{R}_P` and
        :math:`\mathbf{R}_Q`.

        This method recomputes

        .. math::

            \mathbf{R}_P \quad\text{such that}\quad
            \mathbf{P} \approx \mathbf{R}_P^\mathsf{T}\mathbf{R}_P,

        and

        .. math::

            \mathbf{R}_Q \quad\text{such that}\quad
            \mathbf{Q} \approx \mathbf{R}_Q^\mathsf{T}\mathbf{R}_Q,

        using :meth:`_chol_from_cov`. It should be called whenever the
        covariance :math:`\mathbf{P}` or process noise :math:`\mathbf{Q}` may
        have changed externally.

        Notes
        -----
        - :attr:`self.x_hat.cov` is expected to live in the reduced state
          space :math:`\mathbf{y}`, i.e. it is of dimension
          :math:`n \times n` where :math:`n = N-1`.
        - :attr:`self.x_hat.int_cov` is interpreted as the process noise
          covariance :math:`\mathbf{Q}` in the same reduced space.
        """
        self.srcov = self._chol_from_cov(self.x_hat.cov)      # state covariance √P
        self.sric = self._chol_from_cov(self.x_hat.int_cov)   # process noise √Q

    def weighted_cholupdate(self, mat: np.ndarray, vec: np.ndarray, wt: float) -> np.ndarray:
        r"""
        Apply a weighted rank-one update or downdate to a Cholesky factor.

        Let :math:`\mathbf{R} \in \mathbb{R}^{k\times k}` be an upper-triangular
        matrix such that :math:`\mathbf{P} = \mathbf{R}^\mathsf{T}\mathbf{R}`
        represents a covariance. For a vector :math:`\mathbf{v} \in \mathbb{R}^{k}`
        and scalar weight :math:`w`, this function modifies :math:`\mathbf{R}`
        in-place to represent

        .. math::

            \mathbf{P}' =
            \begin{cases}
                \mathbf{P} + w \, \mathbf{v}\mathbf{v}^\mathsf{T}, & w \ge 0, \\
                \mathbf{P} - |w| \, \mathbf{v}\mathbf{v}^\mathsf{T}, & w < 0,
            \end{cases}

        using the :mod:`choldate` routines :func:`cholupdate` and
        :func:`choldowndate`. For negative :math:`w`, this corresponds to
        a covariance **downdate**.

        Parameters
        ----------
        mat : :class:`numpy.ndarray`
            Upper-triangular Cholesky factor :math:`\mathbf{R}` of shape
            ``(k, k)``.
        vec : :class:`numpy.ndarray`
            Update vector :math:`\mathbf{v}`. Can be 1-D of length ``k``,
            or a 2-D array whose rows or columns have length ``k``. In the
            latter case, the rank-one update is applied row-by-row.
        wt : float
            Scalar weight :math:`w`. If :math:`w \ge 0`, a rank-one update
            is performed; if :math:`w < 0`, a downdate is performed using
            :math:`|w|`.

        Returns
        -------
        :class:`numpy.ndarray`
            The updated Cholesky factor :math:`\mathbf{R}'` such that
            :math:`\mathbf{P}' \approx \mathbf{R}'^\mathsf{T}\mathbf{R}'`.

        Raises
        ------
        ValueError
            If the dimension of ``vec`` does not match the size of ``mat``.
        """
        vec = np.array(vec, copy=True)
        shape = vec.shape

        if len(shape) > 2:
            raise ValueError("vec must be 1D or 2D")

        # Handle a stack of vectors row-wise
        if len(shape) == 2 and 1 not in shape:
            r = shape[0]
            if shape[1] != mat.shape[0]:
                # Treat rows vs columns
                vec = vec.T.copy()
                r = shape[1]
            for j in range(r):
                mat = self.weighted_cholupdate(mat, vec[j, :], wt)
            return mat

        v = vec.ravel()
        if v.size != mat.shape[0]:
            raise ValueError("vec length must match Cholesky dimension")

        if wt >= 0.0:
            cholupdate(mat, np.sqrt(wt) * v)
        else:
            choldowndate(mat, np.sqrt(-wt) * v)
        return mat

    # ------------------------------------------------------------------
    # State <-> reduced-state mapping
    # ------------------------------------------------------------------
    def _x_to_y(self, x: np.ndarray) -> np.ndarray:
        r"""
        Map the full quaternion-based state :math:`\mathbf{x}` to the reduced
        UKF state :math:`\mathbf{y}`.

        The full state has the structure

        .. math::

            \mathbf{x}
            =
            \begin{bmatrix}
                \boldsymbol{\omega} \\[3pt]
                \mathbf{q} \\[3pt]
                \mathbf{h}_{\mathrm{RW}} \\[3pt]
                \mathbf{b}_{\mathrm{act}} \\[3pt]
                \mathbf{b}_{\mathrm{sens}} \\[3pt]
                \boldsymbol{\theta}_{\mathrm{dist}}
            \end{bmatrix}
            \in \mathbb{R}^{N},

        with indices:

        - ``x[0:3]``   → :math:`\boldsymbol{\omega} \in \mathbb{R}^3`,
        - ``x[3:7]``   → :math:`\mathbf{q} \in \mathbb{R}^4`,
        - ``x[7:]``    → remaining states
          :math:`[\mathbf{h}_{\mathrm{RW}}, \mathbf{b}_{\mathrm{act}},
          \mathbf{b}_{\mathrm{sens}}, \boldsymbol{\theta}_{\mathrm{dist}}]`.

        The reduced state is

        .. math::

            \mathbf{y}
            =
            \begin{bmatrix}
                \boldsymbol{\omega} \\
                \mathbf{v} \\
                \mathbf{h}_{\mathrm{RW}} \\
                \mathbf{b}_{\mathrm{act}} \\
                \mathbf{b}_{\mathrm{sens}} \\
                \boldsymbol{\theta}_{\mathrm{dist}}
            \end{bmatrix}
            \in \mathbb{R}^{n}, \quad n = N - 1,

        where :math:`\mathbf{v} = f(\mathbf{q}) \in \mathbb{R}^3` is a
        minimal attitude parameterization computed via
        :func:`quat_to_vec3` with mode :attr:`vec_mode`.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Full augmented state vector :math:`\mathbf{x} \in \mathbb{R}^{N}`.

        Returns
        -------
        :class:`numpy.ndarray`
            Reduced state vector :math:`\mathbf{y} \in \mathbb{R}^{N-1}`.
        """
        w = x[0:3]
        q = x[3:7]
        rest = x[7:]
        v3 = quat_to_vec3(q, self.vec_mode)  # 3-vector representation
        return np.concatenate([w, v3, rest])

    def _y_to_x(self, y: np.ndarray) -> np.ndarray:
        r"""
        Map the reduced UKF state :math:`\mathbf{y}` back to the full
        quaternion state :math:`\mathbf{x}`.

        Given

        .. math::

            \mathbf{y}
            =
            \begin{bmatrix}
                \boldsymbol{\omega} \\
                \mathbf{v} \\
                \mathbf{h}_{\mathrm{RW}} \\
                \mathbf{b}_{\mathrm{act}} \\
                \mathbf{b}_{\mathrm{sens}} \\
                \boldsymbol{\theta}_{\mathrm{dist}}
            \end{bmatrix}
            \in \mathbb{R}^{n}, \quad n = N - 1,

        this function reconstructs

        .. math::

            \mathbf{x}
            =
            \begin{bmatrix}
                \boldsymbol{\omega} \\
                \mathbf{q} \\
                \mathbf{h}_{\mathrm{RW}} \\
                \mathbf{b}_{\mathrm{act}} \\
                \mathbf{b}_{\mathrm{sens}} \\
                \boldsymbol{\theta}_{\mathrm{dist}}
            \end{bmatrix}
            \in \mathbb{R}^{N},

        by converting the 3-vector attitude parameter :math:`\mathbf{v}`
        back to a unit quaternion :math:`\mathbf{q}` using
        :func:`vec3_to_quat` and the configured :attr:`vec_mode`.

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            Reduced state vector :math:`\mathbf{y} \in \mathbb{R}^{N-1}`.

        Returns
        -------
        :class:`numpy.ndarray`
            Full augmented state :math:`\mathbf{x} \in \mathbb{R}^{N}`.
        """
        w = y[0:3]
        v3 = y[3:6]
        rest = y[6:]
        q = vec3_to_quat(v3, self.vec_mode)
        return np.concatenate([w, q, rest])

    @staticmethod
    def _flatten_sensors(sensors) -> np.ndarray:
        r"""
        Flatten sensor readings into a single 1-D measurement vector.

        This helper accepts either:

        - a single :class:`numpy.ndarray` containing all stacked measurements, or
        - a list of arrays, each corresponding to a sensor or sensor group.

        In all cases, the return is a 1-D vector

        .. math::

            \mathbf{z}
            \in \mathbb{R}^m,

        where :math:`m` is the total measurement dimension.

        Parameters
        ----------
        sensors : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            Raw sensor readings. Dimension and ordering must be consistent with
            the noise covariance returned by
            :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.sensor_srcov`.

        Returns
        -------
        :class:`numpy.ndarray`
            Flattened measurement vector :math:`\mathbf{z} \in \mathbb{R}^m`.
        """
        if isinstance(sensors, np.ndarray):
            return sensors.ravel()
        return np.concatenate([np.asarray(s).ravel() for s in sensors])

    # ------------------------------------------------------------------
    # Sigma points
    # ------------------------------------------------------------------
    def make_pts_and_wts(self, y0: np.ndarray):
        r"""
        Construct non-augmented sigma points and weights for the reduced state.

        For reduced state dimension :math:`n`, the SR-UKF constructs
        :math:`2n+1` sigma points

        .. math::

            \mathbf{y}^{(0)}, \mathbf{y}^{(1)}, \dots, \mathbf{y}^{(2n)}
            \in \mathbb{R}^{n},

        using the square-root covariance :math:`\mathbf{R}_P` such that
        :math:`\mathbf{P} \approx \mathbf{R}_P^\mathsf{T} \mathbf{R}_P`.

        With parameters :math:`\alpha`, :math:`\kappa`, the scaling factor is

        .. math::

            \lambda = \alpha^2 (n+\kappa) - n, \qquad
            c = n + \lambda, \qquad
            \sqrt{c} = \sqrt{n+\lambda}.

        The sigma points are

        .. math::

            \mathbf{y}^{(0)} = \mathbf{y}_0, \\
            \mathbf{y}^{(i)} = \mathbf{y}_0 + \sqrt{c} \, \mathbf{r}_i,
            \quad i = 1,\dots,n, \\
            \mathbf{y}^{(n+i)} = \mathbf{y}_0 - \sqrt{c} \, \mathbf{r}_i,
            \quad i = 1,\dots,n,

        where :math:`\mathbf{r}_i` is column :math:`i` of :math:`\mathbf{R}_P`.

        The corresponding weights for the mean and covariance are:

        .. math::

            w^{(0)}_m = \frac{\lambda}{c}, \qquad
            w^{(0)}_c = \frac{\lambda}{c} + (1 - \alpha^2 + \beta), \\
            w^{(i)}_m = w^{(i)}_c = \frac{1}{2c}, \quad i = 1, \dots, 2n.

        Parameters
        ----------
        y0 : :class:`numpy.ndarray`
            Reduced mean state :math:`\mathbf{y}_0 \in \mathbb{R}^n`.

        Returns
        -------
        pts : :class:`numpy.ndarray`
            Matrix of sigma points of shape ``(2n+1, n)``.
        w_m : :class:`numpy.ndarray`
            Mean weights of shape ``(2n+1,)``.
        w_c : :class:`numpy.ndarray`
            Covariance weights of shape ``(2n+1,)``.
        """
        R = self.srcov
        n = R.shape[0]
        if n == 0:
            # Degenerate case: no estimated states
            pts = y0.reshape(1, -1)
            w_m = np.array([1.0])
            w_c = np.array([1.0])
            return pts, w_m, w_c

        L = n
        lam = self.al ** 2 * (L + self.kap) - L
        c = L + lam
        scale = np.sqrt(c)

        # Sigma points matrix: (2L+1, n)
        pts = np.zeros((2 * L + 1, n))
        pts[0] = y0

        # Use columns of R as square-root directions
        scaled_R = scale * R
        for i in range(L):
            col = scaled_R[:, i]
            pts[1 + i] = y0 + col
            pts[1 + L + i] = y0 - col

        # Weights
        w_m = np.full(2 * L + 1, 0.5 / c)
        w_c = np.full(2 * L + 1, 0.5 / c)
        w_m[0] = lam / c
        w_c[0] = lam / c + (1.0 - self.al ** 2 + self.bet)

        return pts, w_m, w_c

    # ------------------------------------------------------------------
    # Model hooks (adapt these to your satellite API if needed)
    # ------------------------------------------------------------------
    def _propagate_dynamics(self, x: np.ndarray, u: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Deterministic propagation of the full augmented state over one time step.

        This is a thin wrapper around :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_rk4`,
        which evaluates the spacecraft dynamics (including attitude, reaction
        wheels, and any augmented parameters that appear in the model).

        Given the current full state :math:`\mathbf{x}`, control input
        :math:`\mathbf{u}`, and orbital/environmental state :math:`\mathcal{O}`,
        this function computes

        .. math::

            \mathbf{x}^+ = f(\mathbf{x}, \mathbf{u}, \mathcal{O}, \Delta t),

        where :math:`f` is the numerical integrator implementing the dynamics
        over one step of size :math:`\Delta t`.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Full augmented state :math:`\mathbf{x} \in \mathbb{R}^{N}`.
        u : :class:`numpy.ndarray`
            Control vector :math:`\mathbf{u} \in \mathbb{R}^{n_u}`; its length
            must match :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.control_len`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Current orbital and environmental state, providing quantities such
            as position, velocity, magnetic field, Sun vector, and atmospheric
            density.

        Returns
        -------
        :class:`numpy.ndarray`
            Propagated full state :math:`\mathbf{x}^+ \in \mathbb{R}^{N}`.
        """
        mid_os = self.prev_os.average(orbital_state_2=os)
        return self.est_sat.noiseless_rk4(x=x, u=u, dt=self.dt, orbital_state0=self.prev_os, orbital_state1=os, mid_orbital_state=mid_os)

    def _predict_measurement(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Deterministic prediction of noiseless measurements for a given state.

        This is a wrapper around :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_sensor_readings`.
        For a given full state :math:`\mathbf{x}` and orbital state
        :math:`\mathcal{O}`, it computes

        .. math::

            \mathbf{z} = h(\mathbf{x}, \mathcal{O}),

        where :math:`h` is the (noiseless) satellite measurement model.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Full augmented state :math:`\mathbf{x} \in \mathbb{R}^{N}`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital/environmental state used by the sensor models.

        Returns
        -------
        :class:`numpy.ndarray`
            Predicted measurement vector :math:`\mathbf{z} \in \mathbb{R}^m`,
            flattened as a 1-D array. The dimension :math:`m` must match the
            dimension of the sensor noise covariance returned by
            :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.sensor_srcov`.
        """
        noiseless = self.est_sat.noiseless_sensor_readings(x=x, os=os)
        return noiseless

    # ------------------------------------------------------------------
    # Main UKF step
    # ------------------------------------------------------------------
    def update_core(self, u: np.ndarray, sensors: List[np.ndarray], os: Orbital_State) -> EstimatedArray:
        r"""
        Perform one full Square-root UKF predict–update step.

        This method updates both the **augmented state estimate** and its
        covariance, and synchronizes the associated :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        with the new estimate.

        The update proceeds in the reduced state space
        :math:`\mathbf{y} \in \mathbb{R}^{n}`, but returns the updated
        physical state part :math:`\mathbf{x}_{0: \mathrm{state\_len}}`.

        Steps
        -----
        1. **Refresh square roots**:
           Recompute :math:`\mathbf{R}_P` and :math:`\mathbf{R}_Q` from the
           current covariances :math:`\mathbf{P}` and :math:`\mathbf{Q}`.

        2. **Sigma-point generation**:
           Convert :math:`\mathbf{x}` to :math:`\mathbf{y}`, then build
           :math:`2n+1` sigma points :math:`\mathbf{y}^{(i)}` and weights
           :math:`w^{(i)}_m, w^{(i)}_c`.

        3. **Propagation**:
           For each sigma point:
           
           .. math::

               \mathbf{x}^{(i)} &= \mathrm{map}(\mathbf{y}^{(i)}), \\
               \mathbf{x}^{(i)+} &= f\bigl(\mathbf{x}^{(i)}, \mathbf{u}, \mathcal{O}\bigr), \\
               \mathbf{y}^{(i)+} &= \mathrm{map}^{-1}\bigl(\mathbf{x}^{(i)+}\bigr), \\
               \mathbf{z}^{(i)} &= h\bigl(\mathbf{x}^{(i)+}, \mathcal{O}\bigr),

           where :math:`f` is the dynamics and :math:`h` the measurement model.

        4. **Predicted means**:
           
           .. math::

               \bar{\mathbf{y}} &= \sum_i w^{(i)}_m \mathbf{y}^{(i)+}, \\
               \bar{\mathbf{z}} &= \sum_i w^{(i)}_m \mathbf{z}^{(i)}.

        5. **Time-update covariance**:
           Using the deviations

           .. math::

               \Delta\mathbf{y}^{(i)} = \mathbf{y}^{(i)+} - \bar{\mathbf{y}},

           construct a new square-root covariance
           :math:`\mathbf{R}_P^+` via QR factorization and weighted
           rank-one updates incorporating process noise
           :math:`\mathbf{R}_Q`.

        6. **Measurement covariance and cross-covariance**:
           With measurement deviations
           :math:`\Delta\mathbf{z}^{(i)} = \mathbf{z}^{(i)} - \bar{\mathbf{z}}`,
           build the measurement covariance square root
           :math:`\mathbf{R}_{zz}` and cross-covariance

           .. math::

               \mathbf{P}_{y z}
               = \sum_i w^{(i)}_c
               \Delta\mathbf{y}^{(i)} (\Delta\mathbf{z}^{(i)})^\mathsf{T}.

        7. **Kalman gain**:
           Solve

           .. math::

               \mathbf{K}
               = \mathbf{P}_{y z} \mathbf{P}_{z z}^{-1},

           using triangular solves with :math:`\mathbf{R}_{zz}` such that
           :math:`\mathbf{P}_{z z} \approx \mathbf{R}_{zz}^\mathsf{T}\mathbf{R}_{zz}`.

        8. **Measurement update**:
           Given actual measurements :math:`\mathbf{z}`, the innovation is

           .. math::

               \mathbf{r} = \mathbf{z} - \bar{\mathbf{z}},

           and the posterior mean is

           .. math::

               \mathbf{y}^+ = \bar{\mathbf{y}} + \mathbf{K}\mathbf{r}.

           The posterior covariance is downdated via rank-one updates to
           :math:`\mathbf{R}_P^+` using the product of
           :math:`\mathbf{R}_{zz}` and :math:`\mathbf{K}`.

        9. **Optional cross-term removal** (if ``cross_term == False``):
           Certain cross-covariance blocks between actuator biases, sensor
           biases, and disturbance parameters are set to zero, enforcing a
           block structure of the form:

           .. math::

               \mathbf{P} =
               \begin{bmatrix}
                   * & * & * & * & * \\
                   * & * & * & * & * \\
                   * & * & P_{\mathrm{ab}} & 0 & 0 \\
                   * & * & 0 & P_{\mathrm{sb}} & 0 \\
                   * & * & 0 & 0 & P_{\mathrm{dp}}
               \end{bmatrix},

           where :math:`P_{\mathrm{ab}}`, :math:`P_{\mathrm{sb}}`,
           :math:`P_{\mathrm{dp}}` are the actuator bias, sensor bias,
           and disturbance-parameter subcovariances.

        10. **Commit state and synchronize satellite**:
            The reduced state :math:`\mathbf{y}^+` is mapped back to
            :math:`\mathbf{x}^+`, stored in :attr:`self.x_hat.val` and
            :attr:`self.x_hat.cov`, and then pushed into
            :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite` via :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Control input vector :math:`\mathbf{u} \in \mathbb{R}^{n_u}`.
        sensors : list of :class:`numpy.ndarray` or :class:`numpy.ndarray`
            Measured sensor outputs, stacked in a way consistent with
            :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.sensor_srcov`. The flattened dimension
            is :math:`m`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Current orbital/environmental state.

        Returns
        -------
        :class:`numpy.ndarray`
            Updated **physical spacecraft state** (first
            :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.state_len` entries of :math:`\mathbf{x}^+`),
            i.e.:

            .. math::

                \mathbf{x}^+_{\mathrm{phys}}
                =
                \begin{bmatrix}
                    \boldsymbol{\omega}^+ \\
                    \mathbf{q}^+ \\
                    \mathbf{h}_{\mathrm{RW}}^+
                \end{bmatrix}.
        """
        # ----- 0. Refresh √P and √Q in case covariances changed externally -----
        self._update_square_roots()

        # ----- 1. Build sigma points in reduced state space y -----
        x0 = self.x_hat.val
        y0 = self._x_to_y(x0)                # shape (n,)
        sigma_y, w_m, w_c = self.make_pts_and_wts(y0)
        num_pts, n = sigma_y.shape

        # Storage for propagated states and measurements
        sigma_y_pred = np.zeros_like(sigma_y)
        sigma_z = None

        # ----- 2. Propagate each sigma point through dynamics and measurement -----
        for i in range(num_pts):
            x_i = self._y_to_x(sigma_y[i])
            if i == 5:
                pass
            x_pred = self._propagate_dynamics(x_i, u, os)
            y_pred_i = self._x_to_y(x_pred)

            sigma_y_pred[i] = y_pred_i

            z_i = self._predict_measurement(x_pred, os).ravel()
            if sigma_z is None:
                m_meas = z_i.size
                sigma_z = np.zeros((num_pts, m_meas))
            sigma_z[i, :] = z_i

        # ----- 3. Predicted means -----
        y_mean = np.sum(w_m[:, None] * sigma_y_pred, axis=0)
        z_mean = np.sum(w_m[:, None] * sigma_z, axis=0)

        # Deviations
        Dy = sigma_y_pred - y_mean
        Dz = sigma_z - z_mean

        # ----- 4. Time-update covariance: build new √P via QR + rank-one update -----
        if n == 0:
            P_pred = self.x_hat.cov
            srcov_pred = self.srcov
        else:
            # Exclude the 0th sigma in the QR part; handle it separately with cholupdate
            Dy_scaled = np.sqrt(w_c[1:])[:, None] * Dy[1:]  
            A = np.hstack([Dy_scaled.T, self.sric])         
            # QR of Aᵀ gives upper-triangular R with P ≈ Rᵀ R
            srcov_pred = np.linalg.qr(A.T, mode="r")
            srcov_pred = self.weighted_cholupdate(srcov_pred, Dy[0], w_c[0])
            P_pred = srcov_pred.T @ srcov_pred

        # ----- 5. Measurement covariance √P_zz and cross-covariance P_xy -----
        z_meas = self._flatten_sensors(sensors)
        if z_meas.size != sigma_z.shape[1]:
            raise ValueError(
                f"Measurement dimension mismatch: got {z_meas.size}, "
                f"expected {sigma_z.shape[1]}"
            )

        # Measurement noise square root from satellite model
        sens_srcov = self.est_sat.sensor_srcov()  # shape (m, m)
        m_meas = sens_srcov.shape[0]

        if m_meas != sigma_z.shape[1]:
            raise ValueError(
                f"sensor_srcov() dimension {m_meas} does not match measurement dim {sigma_z.shape[1]}"
            )

        # Predicted measurement covariance via QR
        Dz_scaled = np.sqrt(w_c[1:])[:, None] * Dz[1:]  
        Ay = np.hstack([Dz_scaled.T, sens_srcov])        
        srcov_zz = np.linalg.qr(Ay.T, mode="r")
        srcov_zz = self.weighted_cholupdate(srcov_zz, Dz[0], w_c[0])

        # Cross covariance P_xy (n x m)
        P_xy = np.zeros((n, m_meas))
        for i in range(num_pts):
            P_xy += w_c[i] * np.outer(Dy[i], Dz[i])

        # ----- 6. Kalman gain via triangular solves -----
        # Solve (srcov_zzᵀ srcov_zz) Kᵀ = P_xyᵀ
        # First: srcov_zzᵀ Y = P_xyᵀ  -> Y
        Y_tmp = solve_triangular(srcov_zz.T, P_xy.T, lower=True)
        # Then: srcov_zz Kᵀ = Y      -> Kᵀ
        K_T = solve_triangular(srcov_zz, Y_tmp, lower=False)
        K = K_T.T  # shape (n, m)

        # ----- 7. Measurement update -----
        innov = z_meas - z_mean
        y_post = y_mean + K @ innov

        # Covariance downdate: P⁺ = P⁻ − K P_zz Kᵀ
        # Using square-root downdate with U = R_zz Kᵀ where P_zz = R_zzᵀ R_zz
        U = srcov_zz @ K_T  # shape (m, n)
        srcov_post = self.weighted_cholupdate(srcov_pred, U, -1.0)
        P_post = srcov_post.T @ srcov_post

        # ----- 8. Optional cross-term removal between bias blocks -----
        if not self.cross_term and P_post.size:
            n_rw = self.est_sat.number_RW
            ab0 = 6 + n_rw
            ab1 = ab0 + self.est_sat.act_bias_len
            sb0 = ab1
            sb1 = sb0 + self.est_sat.att_sens_bias_len
            d0 = sb1

            # Zero selected cross-blocks: (act bias) x (sens bias / dist param)
            P_post[ab0:ab1, sb0:sb1] = 0.0
            P_post[sb0:sb1, ab0:ab1] = 0.0
            P_post[ab0:ab1, d0:] = 0.0
            P_post[d0:, ab0:ab1] = 0.0
            P_post[sb0:sb1, d0:] = 0.0
            P_post[d0:, sb0:sb1] = 0.0

            # Re-factor after modification
            srcov_post = self._chol_from_cov(P_post)

        # ----- 9. Commit updated state and covariance -----
        x_post = self._y_to_x(y_post)
        self.x_hat.val = x_post
        self.x_hat.cov = P_post
        # Q_hat (int_cov) stays constant; if you model time-varying Q you can update it here.

        # Push estimate back into the EstimatedSatellite model
        self.est_sat.match_estimate(est_state=self.x_hat, dt=self.dt)

        # Return only the physical satellite state (exclude bias/param states)
        phys_len = self.est_sat.state_len
        phys_val = x_post[:phys_len]
        phys_cov = P_post[:phys_len, :phys_len]

        return EstimatedArray(val=phys_val, cov=phys_cov)