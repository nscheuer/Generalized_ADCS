r"""Equivariant Kalman filter on the spacecraft's physical state group.

The state space is the product Lie group ``SO(3) x R^n``, where the Euclidean
factors include angular rate, wheel momentum, and estimated parameters. Its
right-invariant error uses ``R R_hat.T`` and additive errors elsewhere.
The spacecraft dynamics are lifted to this group and linearized at the
identity of the error, with the configured torques and wheel dynamics included.
Equivariant measurements are linearized at the identity error.

This follows the general EqF construction for systems on Lie groups: a
non-equivariant vector field can be embedded in an equivariant input extension.
The extension is represented here through the error vector field rather than
an explicit basis of the (potentially infinite-dimensional) input space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ADCS.covariance import Covariance
from ADCS.estimators.process_model import propagate_state
from ADCS.estimators.process_noise import (
    assemble_continuous_process_psd,
    control_input_matrix,
    discretize_control_noise,
    van_loan_discretize,
)
from ADCS.helpers.math_helpers import rot_mat, skewsym
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, MTM, StarTracker
from ADCS.state import EstimatorState

from .attitude_estimator import AttitudeEstimator

__all__ = ["EQF"]


class EQF(AttitudeEstimator):
    r"""EqF for the full :class:`~ADCS.state.EstimatorState` layout.

    The error is ``E = X X_hat^{-1}`` in the product group
    ``G = SO(3) x R^n``. Its coordinates are the rotation log and additive
    errors for rate, wheel momentum, biases, and disturbance parameters.
    Prediction uses
    the derivative of the complete deterministic error dynamics at ``E=I``;
    this includes the configured spacecraft torques. Three-axis direction
    measurements and complete three-axis magnetometer sets use their
    equivariant output Jacobian evaluated at the identity error.

    Estimated parameter blocks use translation factors in the group. Their
    deterministic rates are zero, while their configured random-walk noise
    and coupling to the physical dynamics are retained.
    """

    supports_augmented_parameters = True

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
    ) -> None:
        super().__init__(
            satellite,
            state,
            dt=dt,
            covariance_coordinates="tangent",
            correction_mode="rotation_vector",
            measurement_quaternion_mode="rotation_vector",
            quaternion_order="left",
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
        )

    def _group_lift(
        self,
        state: EstimatorState,
        control: np.ndarray,
        orbital_state: Any,
        step: float,
    ) -> np.ndarray:
        r"""Return the right-trivialized spacecraft velocity on ``G``.

        ``R_dot = R [omega]_x`` gives spatial angular velocity ``R omega``.
        The Euclidean group factors use the physical rate and wheel dynamics.
        """
        if state.block_size("estimated_parameters"):
            self.satellite.match_estimate(state, step)
        physical = state.slice("physical", coordinates="full")
        rate = np.asarray(
            self.satellite.dynamics_core(
                state,
                control,
                orbital_state,
                dmode=ErrorMode(
                    add_bias=True,
                    add_noise=False,
                    update_bias=False,
                    update_noise=False,
                ),
            ),
            dtype=float,
        )
        if rate.shape != (physical.stop,):
            raise ValueError(
                f"dynamics_core must return shape ({physical.stop},), got {rate.shape}"
            )
        result = np.zeros(state.tangent_size)
        result[state.slice("angular_velocity", coordinates="tangent")] = rate[:3]
        result[state.slice("attitude", coordinates="tangent")] = rot_mat(state.q) @ state.w
        result[state.slice("wheel_momentum", coordinates="tangent")] = rate[7:]
        return result

    def _error_generator(
        self,
        state: EstimatorState,
        control: np.ndarray,
        orbital_state: Any,
        step: float,
    ) -> np.ndarray:
        r"""Linearize the global group error vector field at the identity.

        For ``E=(Exp(theta), dz)``, the attitude equation is exactly
        ``theta_dot = R_hat domega`` to first order. The Euclidean rows are
        derivatives of the full torque and wheel equations under the same
        group perturbation. Central differences are taken at the fixed error
        identity, avoiding quaternion-coordinate derivatives.
        """
        n = state.tangent_size
        generator = np.zeros((n, n))
        attitude = state.slice("attitude", coordinates="tangent")
        angular_velocity = state.slice("angular_velocity", coordinates="tangent")
        generator[attitude, angular_velocity] = rot_mat(state.q)
        euclidean_rows = np.ones(n, dtype=bool)
        euclidean_rows[attitude] = False
        try:
            for column in range(n):
                difference_step = 1.0e-6
                offset = np.zeros(n)
                offset[column] = difference_step
                plus = state.plus(
                    offset, quaternion_mode="rotation_vector", quaternion_order="left"
                )
                minus = state.plus(
                    -offset, quaternion_mode="rotation_vector", quaternion_order="left"
                )
                derivative = (
                    self._group_lift(plus, control, orbital_state, step)
                    - self._group_lift(minus, control, orbital_state, step)
                ) / (2.0 * difference_step)
                generator[euclidean_rows, column] = derivative[euclidean_rows]
        finally:
            if state.block_size("estimated_parameters"):
                self.satellite.match_estimate(state, step)
        return generator

    def predict(
        self,
        control: Any,
        orbital_state_start: Any,
        orbital_state_end: Any,
        *,
        dt: float | None = None,
        midpoint_orbital_state: Any | None = None,
    ) -> EstimatorState:
        """Propagate the group estimate and its identity-error covariance."""
        self._require_initialized("state to propagate")
        step = self.dt if dt is None else float(dt)
        if not np.isfinite(step) or step < 0.0:
            raise ValueError("dt must be finite and non-negative")
        self._check_step_against_orbit(step, orbital_state_start, orbital_state_end)
        control = np.array(control, dtype=float, copy=True)
        expected_control_shape = (self.satellite.control_len,)
        if control.shape != expected_control_shape:
            raise ValueError(
                f"control must have shape {expected_control_shape}, got {control.shape}"
            )

        prior = self._state
        match_estimate = getattr(self.satellite, "match_estimate", None)
        if match_estimate is not None:
            match_estimate(prior, step)
        generator = self._error_generator(prior, control, orbital_state_start, step)
        continuous_psd = assemble_continuous_process_psd(
            prior,
            self.satellite,
            unmodeled_dynamics_psd=self.unmodeled_dynamics_psd,
            quaternion_mode="rotation_vector",
        )
        transition, process_noise = van_loan_discretize(generator, continuous_psd, step)
        control_covariance = getattr(self.satellite, "control_covariance", None)
        if control_covariance is not None:
            input_matrix = control_input_matrix(
                prior,
                self.satellite,
                control,
                orbital_state_start,
                quaternion_mode="rotation_vector",
                quaternion_order="left",
            )
            process_noise += discretize_control_noise(
                generator, input_matrix, control_covariance().as_matrix(), step
            )
        process_noise = self._add_state_process_noise(process_noise)
        predicted = propagate_state(
            prior,
            self.satellite,
            control,
            step,
            orbital_state_start,
            orbital_state_end,
            midpoint_orbital_state=midpoint_orbital_state,
        )
        predicted_covariance = prior.covariance.predicted_linear(
            transition, process_noise
        )
        predicted.covariance = predicted_covariance
        predicted.process_noise = Covariance(
            process_noise,
            form=prior.process_noise.form,
            coordinates=prior.process_noise.coordinates,
            psd_policy=prior.process_noise.psd_policy,
        )
        self._state = predicted
        self._diagnostics.update(
            error_generator=generator,
            transition=transition,
            process_noise=process_noise,
            predicted_covariance=predicted_covariance.as_matrix(),
        )
        return self.state

    def _transform_innovation(
        self,
        stack: Any,
        active: np.ndarray,
        residual: np.ndarray,
        jacobian: np.ndarray,
        noise: Covariance,
    ) -> tuple[np.ndarray, np.ndarray, Covariance]:
        if not residual.size:
            return residual, jacobian, noise
        rotation = rot_mat(self._state.q)
        transform = np.eye(residual.size)
        magnetic_rows: list[int] = []
        magnetic_axes: list[np.ndarray] = []
        offset = 0
        for entry in stack.selected(active):
            size = entry.residual_size
            if isinstance(entry.source, (StarTracker, EarthHorizonSensor)):
                if size != 3:
                    raise ValueError(
                        f"{entry.name} must have a three-axis direction residual"
                    )
                rows = np.arange(offset, offset + 3)
                transform[np.ix_(rows, rows)] = rotation
            elif isinstance(entry.source, MTM) and size == 1:
                magnetic_rows.append(offset)
                magnetic_axes.append(np.asarray(entry.source.axis, dtype=float))
            offset += size
        if offset != residual.size:
            raise ValueError("active measurement layout does not match the residual")
        # Three independent single-axis magnetometers form a body-frame
        # vector observation. Reconstruct that vector before applying the
        # equivariant inertial-frame output action.
        if len(magnetic_rows) == 3:
            axes = np.vstack(magnetic_axes)
            if np.linalg.cond(axes) < 1.0e8:
                rows = np.asarray(magnetic_rows)
                transform[np.ix_(rows, rows)] = rotation @ np.linalg.inv(axes)
        if np.array_equal(transform, np.eye(residual.size)):
            return residual, jacobian, noise

        residual = transform @ residual
        jacobian = transform @ jacobian
        noise = Covariance(
            transform @ noise.as_matrix() @ transform.T,
            form=noise.form,
            coordinates=noise.coordinates,
            psd_policy=noise.psd_policy,
        )
        return residual, jacobian, noise
