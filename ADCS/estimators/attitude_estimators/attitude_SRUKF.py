"""Square-root, non-augmented unscented Kalman attitude estimator."""

from __future__ import annotations

from typing import Any

from ADCS.state import EstimatorState, State

from .attitude_UKF import UKF


__all__ = ["SRUKF"]


class SRUKF(UKF):
    r"""Non-augmented UKF that stores its state covariance as an upper factor.

    This filter uses the same tangent-state sigma points, manifold means, and
    additive process and measurement noise as :class:`UKF`. Its covariance is
    always retained in ``Covariance(form="sqrt")`` form, so the shared
    unscented covariance operations use square-root QR updates.
    """

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(
            satellite,
            self._square_root_state(state),
            dt=dt,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
            quaternion_mode=quaternion_mode,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

    @staticmethod
    def _square_root_state(state: EstimatorState) -> EstimatorState:
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        result = state.copy()
        result.covariance = state.covariance.copy(form="sqrt")
        result.process_noise = state.process_noise.copy(form="sqrt")
        return result

    def reset(self, state: EstimatorState) -> EstimatorState:
        """Reset while preserving square-root covariance storage."""
        return super().reset(self._square_root_state(state))
