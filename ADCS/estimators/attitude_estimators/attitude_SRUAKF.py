__all__ = ["SRUAKF"]

import numpy as np
import scipy.linalg
import copy
from typing import List, Tuple, Optional
import time

# The external C-library wrapper for fast Cholesky updates
from choldate import cholupdate, choldowndate

from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import SunSensor, SunPair
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import CG5Coefficients
from ADCS.helpers.math_helpers import (
    vec3_to_quat,
    quat_mult,
    normalize,
    state_norm_jac,
)

# Import the base UKF
from ADCS.estimators.attitude_estimators import UAKF

class SRUAKF(UAKF):
    r"""
    Square Root Unscented Kalman Filter (SRUKF) using ``choldate``.

    This class implements a Square Root formulation of the UKF for improved numerical
    stability, ensuring the covariance matrix remains positive semi-definite.
    It inherits dynamics and sigma-point logic from :class:`~ADCS.estimators.attitude_estimators.attitude_UAKF.UAKF`.

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
        Generate augmented sigma points using the stored Square Root S for efficiency.

        Replicates the standard UKF output format (L, pts, wts_m, wts_c, sig0) 
        but uses the persistently tracked self.S for the state block.

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
        pts : list
            List of length 2L+1. Each entry is a list: [full_state, sens_noise, control_noise, int_noise].
        wts_m : numpy.ndarray
            Weights for mean reconstruction.
        wts_c : numpy.ndarray
            Weights for covariance reconstruction.
        sig0 : numpy.ndarray
            The first block of sigma points (state part only), padded.
        """
        # 1. Setup Covariances & Dimensions
        # ---------------------------------
        # State Covariance (Error State)
        # Use self.S directly later, but we need the dimension here.
        L_x = self.S.shape[0]

        # Control/Process Noise Covariance
        control_cov = self.est_sat.control_cov()
        L_q = control_cov.shape[0] if control_cov.size > 0 else 0
        
        # Sensor & Integration Covariances (Zeroed out in this architecture)
        # We maintain their shapes for the list structure
        sens_cov = self.est_sat.sensor_cov(which_sensors=which_sensors)
        L_r = 0 # Explicitly treating as 0 contribution to L
        L_int = 0

        # Total Augmented Dimension
        L = L_x + L_q + L_r + L_int

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
        zeros_int  = np.zeros(self.x_hat.int_cov.shape[0], dtype=dtype) if self.x_hat.int_cov.size > 0 else np.zeros(0, dtype=dtype)

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
        if L_q > 0:
            # We compute Cholesky for Q on the fly (usually small 6x6)
            # Ensure it is Upper Triangular for consistency if needed, 
            # though usually standard Cholesky (Lower) is fine for noise blocks 
            # as long as we take columns. 
            try:
                # np.linalg.cholesky returns Lower.
                L_mat_q = np.linalg.cholesky(control_cov) 
                # Scaled offsets (columns of L_mat_q)
                scaled_L_q = gamma * L_mat_q
                
                # Transpose to get rows for iteration if using similar logic to S
                # or just use columns. Let's use standard: [ +Cols, -Cols ]
                # We need shape (2*L_q, L_q)
                q_offsets = np.hstack((scaled_L_q, -scaled_L_q)).T 
                
                # Append to pts: [mean_state, 0, modified_ctrl, 0]
                for k in q_offsets:
                    pts.append([pt0, zeros_sens, k, zeros_int])
                    
            except np.linalg.LinAlgError:
                # Fallback if Q is not positive definite (rare for process noise)
                pass

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
            full_pre_statej, sens_noise_j, control_noise_j, int_noise_extra_j = pts[j]
            self.sat_match(satj, full_pre_statej)

            post_dyn_state_j = satj.noiseless_rk4(
                x=full_pre_statej[:state_len],
                u=u + control_noise_j,
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
                int_noise_extra_j, 
                post_quat,
            )
            post_pts[j, :] = post_statej.copy()

            self.sat_match(satj, post_full_statej)
            dmode = ErrorMode(add_bias=True, add_noise=False, update_bias=False, update_noise=False)
            sensj = satj.sensor_readings(x=post_full_statej[:state_len],os=os, dmode=dmode)
            post_sens[j, :] = sensj[which_outputs] + sens_noise_j

        # --- 4. Time Update (QR Method) ---
        state1 = wts_m @ post_pts
        dquat1 = vec3_to_quat(state1[3:6], self.vec_mode)
        quat1 = quat_mult(post_quat, dquat1)
        
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

        self.sat_match(satj, state_final)
        return EstimatedArray(val=state_final, cov=P_plus)