"""Invariant-error EKF for attitude estimation.

This implementation uses a left-composed attitude error in the shared
MEKF lifecycle. The spacecraft process and sensor Jacobians remain those of
the configured model, expressed in that error chart.
"""

from __future__ import annotations

from typing import Any

from ADCS.state import EstimatorState, State

from .attitude_MEKF import MEKF

__all__ = ["IEKF"]


class IEKF(MEKF):
    r"""Invariant-error EKF using a right attitude perturbation by default.

    Set ``quaternion_order="left"`` to use the left-composed chart.
    This selects the invariant error convention; it does not assume an
    autonomous/log-linear error model for arbitrary satellite dynamics.
    """

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
        quaternion_order: str = "right",
    ) -> None:
        super().__init__(
            satellite,
            state,
            dt=dt,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        )
