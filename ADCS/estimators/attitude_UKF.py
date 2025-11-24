__all__ = ["UKF"]

import numpy as np
import copy
import scipy
from typing import List

from ADCS.estimators.estimator import Estimator
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import SunSensor, SunPair
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import CG5Coefficients
from ADCS.helpers.math_helpers import (
    quat_to_vec3,
    vec3_to_quat,
    quat_mult,
    quat_inv,
    normalize,
    state_norm_jac,
    matrix_row_normalize,
)


class UKF(Estimator):
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
        r"""
        Unscented Kalman Filter (UKF) with an error-state attitude representation.

        This class implements a UKF for spacecraft attitude determination, handling
        the nonlinearity of attitude dynamics and quaternion normalization. It inherits
        structure from :class:`~ADCS.estimators.estimator.Estimator`.

        The augmented state vector :math:`\mathbf{x}` is managed as follows:

        * **Full State** (``self.x_hat.val``): Always stores the full state including the
            4-component quaternion: :math:`[\boldsymbol{\omega}, \mathbf{q}, \dots]`.
        * **Covariance** (``self.x_hat.cov``): Stores the covariance of the **reduced**
            error state. The structure depends on ``quat_as_vec``:

            * If ``quat_as_vec`` is ``False`` (Standard MEKF):
                The covariance corresponds to :math:`[\delta\boldsymbol{\omega}, \delta\boldsymbol{\theta}, \dots]`,
                where :math:`\delta\boldsymbol{\theta}` is a 3-parameter attitude error representation.
                Dimension is :math:`N-1`.
            * If ``quat_as_vec`` is ``True``:
                The covariance corresponds to :math:`[\delta\boldsymbol{\omega}, \delta\mathbf{q}, \dots]`.
                Dimension is :math:`N`.

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite model defining state structure and dynamics.
        J2000 : float
            Initial time [s].
        x_hat : numpy.ndarray
            Initial full augmented state.
        P_hat : numpy.ndarray
            Initial reduced error-state covariance.
        Q_hat : numpy.ndarray
            Process noise covariance (reduced state).
        dt : float, optional
            Time step [s]. Default 1.0.
        cross_term : bool, optional
            Whether to maintain cross-covariance terms between bias blocks. Default False.
        quat_as_vec : bool, optional
            If True, treats quaternion as a 4-vector in covariance (unconstrained).
            If False, uses error-state 3-vector (standard). Default False.
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
        # UKF tuning parameters
        self.al = 1e-3
        self.kap = 0.0
        self.bet = 2.0
        self.vec_mode = 6

    def determine_covariances_to_use(self, state_cov: np.ndarray, sens_cov: np.ndarray, control_cov: np.ndarray, int_cov: np.ndarray) -> List[bool]:
        r"""
        Determine which noise covariance components to include in the sigma point generation.

        Parameters
        ----------
        state_cov : numpy.ndarray
            State covariance matrix :math:`P`.
        sens_cov : numpy.ndarray
            Sensor noise covariance :math:`R`.
        control_cov : numpy.ndarray
            Control noise covariance.
        int_cov : numpy.ndarray
            Process (integration) noise covariance :math:`Q`.

        Returns
        -------
        list of bool
            A 4-element list indicating inclusion of [state, sensor, control, process] noise.
        """
        use_state_cov = True
        use_sens_cov = False
        use_control_cov = np.size(control_cov)>0 and not np.all(control_cov==0)
        use_int_cov = False
        return [use_state_cov, use_sens_cov, use_control_cov, use_int_cov]

    # ------------------------------------------------------------------ #
    # Sigma-point generation
    # ------------------------------------------------------------------ #
    def make_pts_and_wts(self, pt0: np.ndarray, which_sensors: List[bool]):
        r"""
        Build augmented sigma points and weights.

        Constructs the set of sigma points :math:`\mathcal{X}` based on the current
        state covariance and active noise sources.

        Parameters
        ----------
        pt0 : numpy.ndarray
            Current mean state vector (reduced or full depending on context, usually reduced error mean is zero).
        which_sensors : list of bool
            Mask indicating which sensors are currently active/valid.

        Returns
        -------
        L : int
            Dimension of the augmented error state.
        pts : list
            List of length :math:`2L+1`. Each entry is a list containing:
            ``[full_state, sens_noise, control_noise, int_noise]``.
        wts_m : numpy.ndarray
            Weights for the mean reconstruction (length :math:`2L+1`).
        wts_c : numpy.ndarray
            Weights for the covariance reconstruction (length :math:`2L+1`).
        sig0 : numpy.ndarray
            The first block of sigma points (state part only), padded to shape
            ``(2L+1, len(pt0))``.
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
            mat = np.linalg.cholesky(chol_mat)  # (dim, dim)
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

    # ------------------------------------------------------------------ #
    # Error-state helpers
    # ------------------------------------------------------------------ #
    def reunite_states(
        self,
        dynstate: np.ndarray,
        rest_state: np.ndarray,
        quatref: np.ndarray,
    ) -> np.ndarray:
        r"""
        Reassemble the full state from dynamic and static components, handling quaternion mapping.

        Parameters
        ----------
        dynstate : numpy.ndarray
            The dynamic part of the state (angular rate + attitude).
            Expected to contain the full quaternion.
        rest_state : numpy.ndarray
            The remaining static/bias states.
        quatref : numpy.ndarray
            The reference quaternion used for error-state mapping (if ``quat_as_vec`` is False).

        Returns
        -------
        numpy.ndarray
            The concatenated full state vector in the format required for :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.
        """
        if self.quat_as_vec:
            return np.concatenate((dynstate, rest_state))
        else:
            # dynstate has full attitude quaternion already
            quatdiff = quat_mult(quat_inv(quatref), dynstate[3:7])
            v3diff = quat_to_vec3(quatdiff, self.vec_mode)
            return np.concatenate((dynstate[0:3], v3diff, rest_state))

    def add_to_state(self, state: np.ndarray, add: np.ndarray) -> np.ndarray:
        r"""
        Add a perturbation (error state) to a base state on the manifold.

        This handles the multiplicative update for quaternions:
        :math:`q_{new} = q_{old} \otimes \delta q(\delta \theta)`.

        Parameters
        ----------
        state : numpy.ndarray
            Base state vector(s). Can be 1D or 2D (sigma points).
        add : numpy.ndarray
            Perturbation/Error vector(s). Can be 1D or 2D.

        Returns
        -------
        numpy.ndarray
            The updated state vector(s).
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
        Reconstruct the posterior state after propagation, incorporating integration error noise.

        Parameters
        ----------
        pre_rest_state : numpy.ndarray
            The bias/parameter states before propagation (assumed constant in dynamics).
        post_dynstate : numpy.ndarray
            The propagated dynamic state (rate + quaternion).
        int_err : numpy.ndarray
            Integration error/process noise vector sampled for this sigma point.
        quatref : numpy.ndarray
            Reference quaternion for the sigma point generation.

        Returns
        -------
        post_state : numpy.ndarray
            The reduced error-state representation of the posterior.
        full_state : numpy.ndarray
            The full augmented state representation of the posterior.
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

    # ------------------------------------------------------------------ #
    # Satellite helpers
    # ------------------------------------------------------------------ #
    def sat_match(self, est_sat: EstimatedSatellite, state: np.ndarray) -> None:
        r"""
        Synchronize an EstimatedSatellite instance with a raw state vector.

        Updates ``est_sat`` to match the provided ``state``, ensuring that
        sensor models and dynamics use the specific sigma-point configuration.

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            The satellite model instance to update.
        state : numpy.ndarray
            The full augmented state vector to apply.
        """
        full_statej = self.x_hat.copy()
        full_statej.val[self.use] = state
        est_sat.match_estimate(full_statej, self.dt)

    # ------------------------------------------------------------------ #
    # Main UKF step
    # ------------------------------------------------------------------ #
    def update_core(
        self,
        u: np.ndarray,
        sensors: np.ndarray,   # was effectively used as a NumPy array already
        os: Orbital_State,
    ) -> EstimatedArray:
        r"""
        Perform one UKF predict/update cycle.

        Executes the Unscented Transformation:
        1.  Generates sigma points around the current estimate.
        2.  Propagates sigma points through nonlinear dynamics via
            :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_rk4`.
        3.  Predicts measurements for each sigma point via
            :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.noiseless_sensor_readings`.
        4.  Computes predicted mean, covariance, and Kalman gain.
        5.  Updates the state estimate and covariance using the sensor residuals.

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
            The updated state estimate and reduced covariance matrix.
        """
        u = np.asarray(u, dtype=float).copy()
        os = os.copy()

        state0 = self.x_hat.val.copy()

        # Middle orbital states (CG5 coefficients)
        CG5 = CG5Coefficients()
        # Original code used first 5 coefficients explicitly
        mid_os = [self.prev_os.average(os, CG5.c[j]) for j in range(5)]

        # One-step propagation of the nominal dynamics
        dyn_state0 = self.est_sat.noiseless_rk4(
            x=state0[: self.est_sat.state_len],
            u=u,
            dt=self.dt,
            orbital_state0=self.prev_os,
            orbital_state1=os,
            mid_orbital_state=mid_os,
            quat_as_vec=False,
        )

        # Determine which attitude sensors are active
        which_sensors = [True] * len(self.est_sat.attitude_sensors)
        for j, sensor in enumerate(self.est_sat.attitude_sensors):
            if isinstance(sensor, SunSensor) or isinstance(sensor, SunPair):
                reading = sensor.clean_reading(x=dyn_state0, os=os)
                if np.isnan(reading).any():
                    which_sensors[j] = False

        # Total attitude measurement dimension
        sens_vec_len = sum(
            self.est_sat.sensors[j].output_length
            for j in range(len(self.est_sat.sensors))
            if which_sensors[j]
        )

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
                # j=0 has zero integration noise, so this reference quaternion
                # is clean (same as original implementation).
                post_quat = post_dyn_state_j[3:7].copy()

            post_statej, post_full_statej = self.new_post_state(
                full_pre_statej[state_len:],
                post_dyn_state_j,
                int_noise_extra_j,
                post_quat,
            )
            post_pts[j, :] = post_statej

            self.sat_match(satj, post_full_statej)
            sensj = satj.noiseless_sensor_readings(
                x=post_full_statej[:state_len],
                os=os,
            )
            post_sens[j, :] = sensj[which_sensors]

        # Predicted reduced error state
        state1 = wts_m @ post_pts
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
        y_meas = sensors[which_sensors]
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
        self.sat_match(satj, state2)

        return EstimatedArray(val=state2, cov=cov2)