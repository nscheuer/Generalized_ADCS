r"""Invariant-error extended Kalman filter for spacecraft attitude.

The nominal state is ``(R, omega, h)`` in ``SO(3) x R^(3+n_h)``. With the
default ``quaternion_order='left'``, the attitude error is
``eta_R = R_true R_hat.T`` (right invariant), represented by its Lie log.
Angular velocity and wheel-momentum errors are additive. The spacecraft
dynamics need not be group affine: its error transition is linearized along
the nominal trajectory by the common process model, including chart motion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ADCS.covariance import Covariance
from ADCS.helpers.math_helpers import rot_mat
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, StarTracker
from ADCS.state import EstimatorState

from .attitude_estimator import AttitudeEstimator

__all__ = ["IEKF"]


class IEKF(AttitudeEstimator):
    r"""Lie-group EKF with a right-invariant attitude error by default.

    ``quaternion_order='left'`` gives ``R_true R_hat.T``; choosing ``'right'``
    gives ``R_hat.T R_true``. Both use the SO(3) logarithm and exponential,
    while rates and wheel momentum retain additive errors. The right-error
    mode transports three-axis body-frame direction innovations into the
    inertial frame. General spacecraft torques make the transition depend on
    the nominal trajectory, so the group-affine IEKF autonomy theorem does
    not apply to arbitrary configured disturbances.
    """

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        quaternion_order: str = "left",
    ) -> None:
        super().__init__(
            satellite,
            state,
            dt=dt,
            covariance_coordinates="tangent",
            correction_mode="rotation_vector",
            measurement_quaternion_mode="rotation_vector",
            quaternion_order=quaternion_order,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
        )

    def _transform_innovation(
        self,
        stack: Any,
        active: np.ndarray,
        residual: np.ndarray,
        jacobian: np.ndarray,
        noise: Covariance,
    ) -> tuple[np.ndarray, np.ndarray, Covariance]:
        if self.quaternion_order != "left" or not residual.size:
            return residual, jacobian, noise

        # A body-frame direction y = R.T v has the right-invariant output
        # error R_hat (y_meas - y_hat). Apply its rotation to H and R as well.
        transform = np.eye(residual.size)
        rotation = rot_mat(self._state.q)
        offset = 0
        for entry in stack.selected(active):
            size = entry.residual_size
            if isinstance(entry.source, (StarTracker, EarthHorizonSensor)):
                if size != 3:
                    raise ValueError(f"{entry.name} must have a three-axis direction residual")
                transform[offset:offset + 3, offset:offset + 3] = rotation
            offset += size
        if offset != residual.size:
            raise ValueError("active measurement layout does not match the residual")
        if np.array_equal(transform, np.eye(residual.size)):
            return residual, jacobian, noise
        covariance = transform @ noise.as_matrix() @ transform.T
        return (
            transform @ residual,
            transform @ jacobian,
            Covariance(
                covariance,
                form=noise.form,
                coordinates=noise.coordinates,
                psd_policy=noise.psd_policy,
            ),
        )
