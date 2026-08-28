"""Shared continuous-time process-noise construction for attitude estimators.

The functions in this module deliberately operate on ``EstimatorState``'s
named layout.  They are filter-neutral: an EKF can use the returned error-state
Jacobian directly, while any future sigma-point estimator can reuse the same
Van Loan covariance discretization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

from ADCS.state import EstimatorState, State


__all__ = [
    "assemble_continuous_process_psd",
    "error_state_transfer",
    "continuous_error_state_model",
    "van_loan_discretize",
    "discretize_process_noise",
]


def _as_psd(value: Any, size: int, *, name: str) -> np.ndarray:
    """Normalize a scalar, diagonal vector, or square PSD to a matrix."""
    if size == 0:
        array = np.asarray(value, dtype=float)
        if array.size not in (0, 1) or (array.size == 1 and array.item() != 0.0):
            raise ValueError(f"{name} cannot describe an empty state block")
        return np.zeros((0, 0))
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        if not np.isfinite(array) or array < 0.0:
            raise ValueError(f"{name} scalar must be finite and non-negative")
        return np.eye(size) * float(array)
    if array.ndim == 1:
        if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} vector must have {size} finite, non-negative entries")
        return np.diag(array)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be scalar, length-{size} vector, or ({size}, {size}) matrix")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} matrix must be symmetric")
    if np.linalg.eigvalsh(array)[0] < -1e-12:
        raise ValueError(f"{name} matrix must be positive semidefinite")
    return (array + array.T) / 2.0


def _hardware_psd(satellite: Any, method: str, size: int, *, name: str) -> np.ndarray:
    try:
        value = getattr(satellite, method)().as_matrix()
    except AttributeError as error:
        raise TypeError(
            f"satellite must provide {method}() to assemble {name} process noise"
        ) from error
    value = np.asarray(value, dtype=float)
    if value.shape != (size, size):
        raise ValueError(
            f"{method}() returned {value.shape}, but EstimatorState {name} block has size {size}"
        )
    return value


def assemble_continuous_process_psd(
    state: EstimatorState,
    satellite: Any,
    *,
    unmodeled_dynamics_psd: Any = 0.0,
    quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
) -> np.ndarray:
    r"""Assemble chart-local continuous PSD :math:`Q_c` from the state layout.

    ``unmodeled_dynamics_psd`` applies to the physical local state
    ``[angular_velocity, attitude, wheel_momentum]`` and accepts a scalar, a
    diagonal vector, or a full PSD matrix. Bias and disturbance blocks are
    populated from their configured hardware random-walk rates. The returned
    coordinates match ``quaternion_mode``.
    """
    if not isinstance(state, EstimatorState):
        raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")

    full_quaternion = quaternion_mode == "full_quaternion"
    if full_quaternion:
        size = state.full_size
        slices = state.full_slices
        physical = slice(0, slices["wheel_momentum"].stop)
    else:
        # This also validates aliases and unsupported chart names.
        state.tangent_map(quaternion_mode=quaternion_mode)
        size = state.tangent_size
        slices = state.tangent_slices
        physical = slices["physical"]
    result = np.zeros((size, size), dtype=float)
    result[physical, physical] += _as_psd(
        unmodeled_dynamics_psd,
        physical.stop - physical.start,
        name="unmodeled_dynamics_psd",
    )
    for block, method in (
        ("actuator_bias", "actuator_bias_process_psd"),
        ("sensor_bias", "sensor_bias_process_psd"),
        ("disturbance_parameter", "disturbance_parameter_process_psd"),
    ):
        sl = slices[block]
        result[sl, sl] += _hardware_psd(satellite, method, sl.stop - sl.start, name=block)
    if full_quaternion:
        normalizer = state.normalization_jacobian()
        result = normalizer @ result @ normalizer.T
    return (result + result.T) / 2.0


def error_state_transfer(
    state: EstimatorState,
    satellite: Any,
    control: np.ndarray,
    orbital_state: Any,
    *,
    quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    quaternion_order: str = State.DEFAULT_QUATERNION_ORDER,
) -> np.ndarray:
    r"""Return the continuous local error-state transfer matrix ``F``.

    ``F`` is chart-local at ``state``; it must be rebuilt after the estimate's
    linearization point or quaternion chart changes.

    ``EstimatedSatellite.dynJacCore`` stores derivative variables in rows and
    derivative outputs in columns.  This function converts that historical
    convention to the conventional column-error matrix before augmenting it
    with random-walk states and reducing quaternion coordinates through
    ``EstimatorState.tangent_map`` and ``tangent_pinv``.
    """
    if not isinstance(state, EstimatorState):
        raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
    blocks = satellite.dynJacCore(state, np.asarray(control, dtype=float), orbital_state)
    if len(blocks) != 5:
        raise ValueError("dynJacCore must return five state, control, and estimated-parameter blocks")
    dxdot_dx, _, dxdot_dab, dxdot_dsb, dxdot_ddp = blocks
    full = np.zeros((state.full_size, state.full_size), dtype=float)
    base = state.full_slices["wheel_momentum"].stop
    if np.asarray(dxdot_dx).shape != (base, base):
        raise ValueError(f"dynJacCore state block must have shape {(base, base)}")
    full[:base, :base] = np.asarray(dxdot_dx, dtype=float).T
    for name, block in (
        ("actuator_bias", dxdot_dab),
        ("sensor_bias", dxdot_dsb),
        ("disturbance_parameter", dxdot_ddp),
    ):
        sl = state.full_slices[name]
        expected = (sl.stop - sl.start, base)
        if np.asarray(block).shape != expected:
            raise ValueError(f"dynJacCore {name} block must have shape {expected}")
        full[:base, sl] = np.asarray(block, dtype=float).T
    if quaternion_mode == "full_quaternion":
        normalizer = state.normalization_jacobian()
        return normalizer @ full @ normalizer
    tangent = state.tangent_map(
        quaternion_mode=quaternion_mode, quaternion_order=quaternion_order
    )
    tangent_pinv = state.tangent_pinv(
        quaternion_mode=quaternion_mode, quaternion_order=quaternion_order
    )
    return tangent_pinv @ full @ tangent


def continuous_error_state_model(
    state: EstimatorState,
    satellite: Any,
    control: np.ndarray,
    orbital_state: Any,
    *,
    unmodeled_dynamics_psd: Any = 0.0,
    quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    quaternion_order: str = State.DEFAULT_QUATERNION_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Build shared continuous-time ``(F, Qc)`` for an estimator prediction."""
    return (
        error_state_transfer(
            state,
            satellite,
            control,
            orbital_state,
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        ),
        assemble_continuous_process_psd(
            state,
            satellite,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
            quaternion_mode=quaternion_mode,
        ),
    )


def van_loan_discretize(
    transfer: Any,
    continuous_psd: Any,
    dt: float,
    *,
    noise_input: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Discretize ``xdot = F x + L w`` with Van Loan's matrix exponential.

    ``continuous_psd`` is ``Q_c`` in noise coordinates; omitting
    ``noise_input`` uses ``L=I`` and therefore accepts state-space PSDs
    assembled by :func:`assemble_continuous_process_psd`.
    """
    transfer = np.asarray(transfer, dtype=float)
    if transfer.ndim != 2 or transfer.shape[0] != transfer.shape[1]:
        raise ValueError("transfer must be square")
    n = transfer.shape[0]
    continuous_psd = np.asarray(continuous_psd, dtype=float)
    if noise_input is None:
        noise_input = np.eye(n)
    noise_input = np.asarray(noise_input, dtype=float)
    if noise_input.ndim != 2 or noise_input.shape[0] != n:
        raise ValueError(f"noise_input must have {n} rows")
    m = noise_input.shape[1]
    if continuous_psd.shape != (m, m):
        raise ValueError(f"continuous_psd must have shape {(m, m)}")
    if not np.all(np.isfinite(transfer)) or not np.all(np.isfinite(continuous_psd)):
        raise ValueError("transfer and continuous_psd must be finite")
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")

    diffusion = noise_input @ continuous_psd @ noise_input.T
    van_loan = np.zeros((2 * n, 2 * n), dtype=float)
    van_loan[:n, :n] = -transfer
    van_loan[:n, n:] = diffusion
    van_loan[n:, n:] = transfer.T
    exponential = expm(van_loan * dt)
    transition = exponential[n:, n:].T
    discrete_psd = transition @ exponential[:n, n:]
    return transition, (discrete_psd + discrete_psd.T) / 2.0


def discretize_process_noise(
    state: EstimatorState,
    satellite: Any,
    control: np.ndarray,
    orbital_state: Any,
    dt: float,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and Van Loan-discretize the shared attitude process-noise model.

    The returned transition :math:`\\Phi` is chart-local at ``state``.
    """
    transfer, continuous_psd = continuous_error_state_model(
        state, satellite, control, orbital_state, **kwargs
    )
    return van_loan_discretize(transfer, continuous_psd, dt)
