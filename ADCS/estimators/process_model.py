"""Pure estimator-facing state propagation.

This module is the narrow boundary around the satellite's historical
integration API. Filters call :func:`propagate_state`; they do not need to know
how ``noiseless_rk4`` represents physical versus augmented estimator state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ADCS.state import EstimatorState, State


__all__ = ["propagate_state"]


def propagate_state(
    state: State,
    satellite: Any,
    control: Any,
    dt: float,
    orbital_state_start: Any,
    orbital_state_end: Any,
    *,
    midpoint_orbital_state: Any | None = None,
    quaternion_integrator: str = "rk4",
) -> State:
    r"""Return deterministic propagation without mutating ``state``.

    The physical state is propagated by ``satellite.noiseless_rk4``. For an
    :class:`EstimatorState`, wheel momentum is propagated while estimated bias
    and disturbance blocks, covariance, and process noise are copied unchanged.
    Those nominal parameter blocks are constant under this deterministic model;
    their uncertainty evolves separately through the process-noise model.

    ``quaternion_integrator`` is ``"rk4"`` for normalized component RK4 or
    ``"cg5"`` for the satellite's Lie-group variant.
    """
    if not isinstance(state, State):
        raise TypeError(f"state must be a State, got {type(state).__name__}")
    control = np.array(control, dtype=float, copy=True)
    if control.ndim != 1:
        raise ValueError(f"control must be one-dimensional, got shape {control.shape}")
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if quaternion_integrator not in ("rk4", "cg5"):
        raise ValueError("quaternion_integrator must be 'rk4' or 'cg5'")

    expected_size = getattr(satellite, "state_len", state.full_size)
    physical_size = state.slice("physical", coordinates="full").stop
    if physical_size != expected_size:
        raise ValueError(
            f"state physical block has size {physical_size}, "
            f"but satellite expects {expected_size}"
        )

    integrator_midpoint = midpoint_orbital_state
    if quaternion_integrator == "cg5":
        if isinstance(midpoint_orbital_state, Sequence):
            if len(midpoint_orbital_state) != 5:
                raise ValueError("cg5 midpoint_orbital_state must contain five stage states")
        elif midpoint_orbital_state is not None:
            integrator_midpoint = None

    propagated = satellite.noiseless_rk4(
        state,
        control,
        dt,
        orbital_state_start,
        orbital_state_end,
        verbose=False,
        mid_orbital_state=integrator_midpoint,
        quat_as_vec=quaternion_integrator == "rk4",
        give_err_est=False,
    )
    if not isinstance(propagated, State):
        raise TypeError(
            "satellite.noiseless_rk4() must return a State when error estimation is disabled"
        )

    if not isinstance(state, EstimatorState):
        return propagated

    result = state.copy()
    result.w = propagated.w
    result.q = propagated.q
    result.h = propagated.h
    return result
