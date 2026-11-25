"""Shared input contract for compiled attitude estimators.

The Python object validates initial data once. Simulation loops use only its
numeric arrays and the compiled kernels supplied by a concrete estimator.
"""

import numpy as np


class AttitudeEstimator:
    """Nominal body-from-inertial attitude, body rate, and 6D error covariance.

    Error coordinates are ``[body-frame rotation vector, body-rate error]``.
    No sensor or actuator bias is part of this state.
    """

    DIAGNOSTIC_FIELDS = frozenset(
        (
            "estimated_attitude",
            "estimated_rate",
            "estimated_attitude_covariance",
            "estimated_rate_covariance",
            "measurement_innovation",
            "measurement_innovation_covariance",
            "normalized_innovation_squared",
            "normalized_estimation_error_squared",
            "measurement_residual",
            "kalman_gain",
        )
    )

    def __init__(
        self,
        quaternion: np.ndarray,
        rate: np.ndarray,
        covariance: np.ndarray,
        acceleration_psd: float | np.ndarray,
        *,
        record: tuple[str, ...] = (),
    ) -> None:
        q = np.asarray(quaternion, dtype=np.float64)
        w = np.asarray(rate, dtype=np.float64)
        p = np.asarray(covariance, dtype=np.float64)
        a = np.asarray(acceleration_psd, dtype=np.float64)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) == 0:
            raise ValueError("quaternion must be a finite nonzero Hamilton 4-vector")
        if w.shape != (3,) or not np.all(np.isfinite(w)):
            raise ValueError("rate must be a finite 3-vector")
        if p.shape != (6, 6) or not np.all(np.isfinite(p)) or not np.allclose(p, p.T):
            raise ValueError("covariance must be a finite symmetric 6x6 matrix")
        if np.linalg.eigvalsh(p).min() < -1e-12:
            raise ValueError("covariance must be positive semidefinite")
        if a.ndim == 0:
            a = np.eye(3) * float(a)
        if a.shape != (3, 3) or not np.all(np.isfinite(a)) or not np.allclose(a, a.T):
            raise ValueError("acceleration_psd must be a finite symmetric 3x3 matrix")
        if np.linalg.eigvalsh(a).min() < -1e-12:
            raise ValueError("acceleration_psd must be positive semidefinite")
        self.quaternion = np.ascontiguousarray(q / np.linalg.norm(q))
        self.rate = np.ascontiguousarray(w)
        self.covariance = np.ascontiguousarray(p)
        self.acceleration_psd = np.ascontiguousarray(a)
        self.record = frozenset(record)
        unknown = self.record - self.DIAGNOSTIC_FIELDS
        if unknown:
            raise ValueError(f"unknown estimator diagnostic fields: {sorted(unknown)}")

    def prepare(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return independent numeric state for one simulation run."""
        return (
            self.quaternion.copy(),
            self.rate.copy(),
            self.covariance.copy(),
            self.acceleration_psd.copy(),
        )
