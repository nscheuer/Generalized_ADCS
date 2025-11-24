__all__ = ["SRUKF"]

import numpy as np
import scipy.linalg
import copy
from typing import List, Tuple, Optional

# The external C-library wrapper for fast Cholesky updates
from choldate import cholupdate, choldowndate

from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import SunSensor, SunPair
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import CG5Coefficients
from ADCS.helpers.math_helpers import (
    vec3_to_quat,
    quat_mult,
    normalize,
    state_norm_jac,
)

# Import the base UKF
from ADCS.estimators.attitude_UKF import UKF 

class SRUKF(UKF):
    r"""
    Square Root Unscented Kalman Filter (SRUKF) using ``choldate``.

    This class implements a Square Root formulation of the UKF for improved numerical
    stability, ensuring the covariance matrix remains positive semi-definite.
    It inherits dynamics and sigma-point logic from :class:`~ADCS.estimators.attitude_UKF.UKF`.

    Unlike the standard UKF which propagates the full covariance matrix :math:`P`,
    this estimator propagates the **Upper Triangular** Cholesky factor :math:`S`,
    defined such that:

    .. math::

        P = S^\top S

    Crucially, this implementation utilizes the ``choldate`` C-extension for fast
    rank-1 updates and downdates, which is essential for handling the negative
    weight associated with the central sigma point (:math:`W_0^{(c)} < 0`).

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Estimated satellite model defining state structure and dynamics.
    J2000 : float
        Initial time [s].
    x_hat : numpy.ndarray
        Initial full augmented state.
    P_hat : numpy.ndarray
        Initial reduced error-state covariance. The square root :math:`S`
        is computed from this upon initialization.
    Q_hat : numpy.ndarray
        Process noise covariance (reduced state). The square root :math:`S_Q`
        is computed from this.
    dt : float, optional
        Time step [s]. Default 1.0.
    cross_term : bool, optional
        Whether to maintain cross-covariance terms between bias blocks. Default False.
    quat_as_vec : bool, optional
        If True, treats quaternion as a 4-vector in covariance. Default False.
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
    ) -> None:
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

        # Initialize State Covariance Square Root (Upper Triangular)
        # Try Cholesky first (requires P > 0)
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
        Wrapper for ``choldate`` handling weighted updates and matrix-row operations.

        Performs a rank-1 update or downdate on the Cholesky factor ``mat`` based on
        the sign of the weight ``wt``.

        .. math::

            S_{new}^\top S_{new} = S^\top S \pm \sqrt{|wt|} \mathbf{v} \mathbf{v}^\top

        This method handles:
        1.  **Sign Logic:** Calls :func:`choldate.cholupdate` if :math:`wt \ge 0` and
            :func:`choldate.choldowndate` if :math:`wt < 0`.
        2.  **Matrix Inputs:** If ``vec`` is a matrix, it recursively applies updates
            for each row/column vector.

        Parameters
        ----------
        mat : numpy.ndarray
            The Upper Triangular Cholesky factor :math:`S`.
        vec : numpy.ndarray
            The vector (or matrix of vectors) to update with.
        wt : float
            The weight of the update. Negative weights trigger a downdate.

        Returns
        -------
        numpy.ndarray
            The updated Upper Triangular matrix.
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

    def make_pts_and_wts(self, pt0: np.ndarray, which_sensors: List[bool]):
        r"""
        Generate sigma points directly using the stored Upper Triangular :math:`S`.

        Overrides :meth:`~ADCS.estimators.attitude_UKF.UKF.make_pts_and_wts` to avoid
        performing a Cholesky decomposition at every step. Instead, it scales the
        persistently tracked :math:`S` matrix.

        Parameters
        ----------
        pt0 : numpy.ndarray
            Current mean state vector.
        which_sensors : list of bool
            Mask indicating active sensors.

        Returns
        -------
        L : int
            Dimension of the augmented error state.
        pts : numpy.ndarray
            Array of sigma points.
        wts_m : numpy.ndarray
            Mean weights.
        wts_c : numpy.ndarray
            Covariance weights.
        """
        L_dim = self.state_len - 1 if not self.quat_as_vec else self.state_len
        
        self.lam = self.al ** 2.0 * (self.kap + L_dim) - L_dim
        gamma = np.sqrt(L_dim + self.lam)

        # Scale the stored Upper Triangular factor
        weighted_S = gamma * self.S

        # Generate offsets
        # Since S is Upper, the rows are the conjugate axes.
        # We stack [S, -S]
        offsets = np.vstack((weighted_S, -weighted_S))

        # Add to mean state (handle quaternions via UKF base method)
        states = self.add_to_state(pt0, offsets)
        
        # Full Sigma Points: [Mean, +Sigmas, -Sigmas]
        pts = np.vstack((pt0, states))

        # Weights
        denom = L_dim + self.lam
        w0_m = self.lam / denom
        w0_c = self.lam / denom + (1.0 - self.al ** 2.0 + self.bet)
        wi = 0.5 / denom

        num_sigma = 2 * L_dim + 1
        wts_m = np.full(num_sigma, wi)
        wts_c = np.full(num_sigma, wi)
        wts_m[0] = w0_m
        wts_c[0] = w0_c
        
        return L_dim, pts, wts_m, wts_c

    def update_core(
        self,
        u: np.ndarray,
        sensors: np.ndarray,
        os: Orbital_State,
    ) -> EstimatedArray:
        r"""
        Perform one SRUKF predict/update cycle using QR decomposition.

        This method implements the Square Root UKF algorithm:

        1.  **Prediction**:
            
            * Propagates sigma points via :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_rk4`.
            * Computes the *a priori* Cholesky factor :math:`S^-` using QR decomposition 
                of the weighted sigma-point deviations and process noise root :math:`S_Q`.
            * Applies a rank-1 downdate for the 0-th sigma point (negative weight).

        2.  **Measurement**:
            
            * Predicts measurements and computes :math:`S_{yy}` (innovation covariance root)
                using QR decomposition of measurement deviations and sensor noise root :math:`S_R`.
            * Computes Kalman Gain :math:`K` using forward/backward substitution.

        3.  **Update**:
            
            * Updates state mean.
            * Updates Cholesky factor :math:`S` via sequential Cholesky downdates 
                using the columns of :math:`U = S_{yy} K^\top`.

        Parameters
        ----------
        u : numpy.ndarray
            Control input vector.
        sensors : numpy.ndarray
            Stacked array of actual sensor measurements.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Current orbital environment state.

        Returns
        -------
        :class:`~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedArray`
            The updated state estimate and reduced covariance matrix (reconstructed from :math:`S`).
        """
        u = np.asarray(u, dtype=float).copy()
        os = os.copy()
        state0 = self.x_hat.val.copy()

        # --- 1. Determine Active Sensors (Eclipse Check) ---
        CG5 = CG5Coefficients()
        mid_os = [self.prev_os.average(os, CG5.c[j]) for j in range(5)]
        
        # Nominal propagation
        dyn_state0 = self.est_sat.noiseless_rk4(
            x=state0[:self.est_sat.state_len],
            u=u,
            dt=self.dt,
            orbital_state0=self.prev_os,
            orbital_state1=os,
            mid_orbital_state=mid_os,
            quat_as_vec=False,
        )

        which_sensors = [True] * len(self.est_sat.attitude_sensors)
        for j, sensor in enumerate(self.est_sat.attitude_sensors):
            if isinstance(sensor, (SunSensor, SunPair)):
                reading = sensor.clean_reading(x=dyn_state0, os=os)
                if np.isnan(reading).any():
                    which_sensors[j] = False

        # --- 2. Sigma Point Generation ---
        sens_vec_len = sum(
            self.est_sat.sensors[j].output_length
            for j in range(len(self.est_sat.sensors))
            if which_sensors[j]
        )

        L_dim, pts, wts_m, wts_c = self.make_pts_and_wts(state0, which_sensors)
        num_sigma = pts.shape[0]
        
        sigma_state_len = state0.size - 1 + self.quat_as_vec
        post_pts = np.empty((num_sigma, sigma_state_len), dtype=float)
        post_sens = np.empty((num_sigma, sens_vec_len), dtype=float)

        est_sat_template = self.est_sat
        state_len = est_sat_template.state_len
        satj = copy.deepcopy(est_sat_template)
        post_quat = None

        # --- 3. Propagation Loop ---
        for j in range(num_sigma):
            full_pre_statej = pts[j]
            self.sat_match(satj, full_pre_statej)

            post_dyn_state_j = satj.noiseless_rk4(
                x=full_pre_statej[:state_len],
                u=u,
                dt=self.dt,
                orbital_state0=self.prev_os,
                orbital_state1=os,
                mid_orbital_state=mid_os,
                quat_as_vec=False,
            )

            if j == 0:
                post_quat = post_dyn_state_j[3:7].copy()

            post_statej, post_full_statej = self.new_post_state(
                full_pre_statej[state_len:],
                post_dyn_state_j,
                np.zeros(sigma_state_len), 
                post_quat,
            )
            post_pts[j, :] = post_statej

            self.sat_match(satj, post_full_statej)
            sensj = satj.noiseless_sensor_readings(
                x=post_full_statej[:state_len],
                os=os,
            )
            post_sens[j, :] = sensj[which_sensors]

        # --- 4. Time Update (QR Method) ---
        state1 = wts_m @ post_pts
        dquat1 = vec3_to_quat(state1[3:6], self.vec_mode)
        quat1 = quat_mult(post_quat, dquat1)
        
        # State deviations
        pts_diff = post_pts - state1
        
        # QR Decomposition for State Covariance
        # A = [ sqrt(w) * (Xi - x).T | sqrt(Q) ]
        # We need R from QR(A).
        # Note: wts_c[0] is negative, so we skip index 0 in the QR and downdate it later.
        
        weighted_diffs = pts_diff[1:, :].T * np.sqrt(wts_c[1:])
        
        # Stack weighted sigmas and process noise (S_Q is Upper)
        # QR of transpose gives R upper triangular
        # A = np.hstack([weighted_diffs, self.S_Q])
        # R = qr(A.T, mode='r')  -> This is wrong if S_Q is Upper.
        # We want R such that R.T R = A A.T
        # A.T = vstack([weighted_diffs.T, self.S_Q.T])? No.
        
        # Correct logic from old code:
        # srcov1 = qr( hstack([ diffs.T, S_Q ]).T )
        #      = qr( vstack([ diffs, S_Q.T ]) ) -> This seems to mix Upper/Lower?
        
        # Let's strictly follow the old code: 
        # srcov1 = np.linalg.qr(np.hstack([pts_diff[1:,:].T*np.sqrt(wts_c[1:]), self.sric]).T, mode='r')
        # Here sric (self.S_Q) was initialized as Upper.
        # hstack puts them side by side. .T flips it.
        # So we represent the matrix A where A^T A = P. 
        
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
        
        # Solve for Gain
        # Kk = (Pyy^-1 Pyx)^T
        # Pyy = srcov_sens.T @ srcov_sens (since srcov_sens is Upper R, P = R.T R)
        
        # Old code solution:
        # Kk = solve_triangular(srcov_sens, solve_triangular(srcov_sens, covyx, trans='T'))
        # Let's verify: 
        # Inner: x = solve(R^T, P_yx).  R^T x = P_yx.
        # Outer: K = solve(R, x).       R K = x.
        # R K = R^-T P_yx  ->  R^T R K = P_yx  -> P_yy K = P_yx. Correct.
        
        try:
            t1 = scipy.linalg.solve_triangular(srcov_sens, covyx, lower=False, trans='T')
            Kk = scipy.linalg.solve_triangular(srcov_sens, t1, lower=False)
        except np.linalg.LinAlgError:
             raise np.linalg.LinAlgError('SRUKF: Singular Matrix in Update')

        # --- 7. State Update ---
        # Note: Kk here is (n_meas, n_state).
        # State row vec update: x_new = x + (y - y_est) @ Kk
        y_meas = sensors[which_sensors]
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

        self.sat_match(satj, state_final)
        
        return EstimatedArray(val=state_final, cov=P_plus)