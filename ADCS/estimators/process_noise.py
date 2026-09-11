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

from ADCS.satellite_hardware.errors import ErrorMode
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


def _tangent_map_rate(
    state: EstimatorState,
    state_rate: np.ndarray,
    *,
    quaternion_mode: str,
    quaternion_order: str,
) -> np.ndarray:
    r"""Return :math:`\dot G` for the local-to-full tangent map.

    The tangent map depends on the nominal quaternion.  A local perturbation
    satisfies ``delta_full = G(q) @ delta_local``; differentiating this
    relation gives ``delta_dot_local = G^dagger (A G - G_dot) delta_local``.
    ``G_dot`` is evaluated along the nominal quaternion trajectory.  Keeping
    this small numerical derivative here supports every quaternion chart
    already implemented by :class:`State` without duplicating chart-specific
    derivatives.
    """
    state_rate = np.asarray(state_rate, dtype=float)
    if state_rate.shape != (state.full_size,):
        raise ValueError(
            f"state_rate must have shape ({state.full_size},), got {state_rate.shape}"
        )
    q_rate = state_rate[state.slice("attitude", coordinates="full")]
    tangent = state.tangent_map(
        quaternion_mode=quaternion_mode,
        quaternion_order=quaternion_order,
    )
    if np.linalg.norm(q_rate) == 0.0:
        return np.zeros_like(tangent)

    step = 1.0e-7
    plus = state.copy()
    minus = state.copy()
    plus.q = state.q + step * q_rate
    minus.q = state.q - step * q_rate
    return (
        plus.tangent_map(
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        )
        - minus.tangent_map(
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        )
    ) / (2.0 * step)


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
        slices = state.slices(coordinates="full")
        physical = slices["physical"]
    else:
        # This also validates aliases and unsupported chart names.
        state.tangent_map(quaternion_mode=quaternion_mode)
        size = state.tangent_size
        slices = state.slices(coordinates="tangent")
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
    source_quaternion_mode: str | None = None,
    target_quaternion_mode: str | None = None,
    quaternion_order: str = State.DEFAULT_QUATERNION_ORDER,
) -> np.ndarray:
    r"""Return the continuous local error-state transfer matrix ``F``.

    By default the source and target charts are both ``quaternion_mode``.
    ``source_quaternion_mode`` and ``target_quaternion_mode`` may be supplied
    independently to build rectangular coordinate-transition Jacobians such as
    tangent-to-full-quaternion or full-quaternion-to-tangent.  The returned
    matrix maps a perturbation expressed in the source chart to a perturbation
    rate expressed in the target chart, evaluated at ``state``.

    Because the target tangent map moves with the nominal attitude, the matrix
    includes the target chart-motion term ``G_dot``.  For matching source and
    target charts this reduces to the familiar ``G^\dagger(A G - \dot G)``.

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
        raise ValueError(
            "dynJacCore must return five state, control, and estimated-parameter blocks"
        )
    dxdot_dx, _, dxdot_dab, dxdot_dsb, dxdot_ddp = blocks
    full = np.zeros((state.full_size, state.full_size), dtype=float)
    slices = state.slices(coordinates="full")
    base = slices["physical"].stop
    if np.asarray(dxdot_dx).shape != (base, base):
        raise ValueError(f"dynJacCore state block must have shape {(base, base)}")
    full[:base, :base] = np.asarray(dxdot_dx, dtype=float).T
    for name, block in (
        ("actuator_bias", dxdot_dab),
        ("sensor_bias", dxdot_dsb),
        ("disturbance_parameter", dxdot_ddp),
    ):
        sl = slices[name]
        expected = (sl.stop - sl.start, base)
        if np.asarray(block).shape != expected:
            raise ValueError(f"dynJacCore {name} block must have shape {expected}")
        full[:base, sl] = np.asarray(block, dtype=float).T

    dynamics_core = getattr(satellite, "dynamics_core", None)
    if dynamics_core is None:
        # Keep lightweight Jacobian providers usable.  Real spacecraft models
        # provide dynamics_core, which is required for the moving-chart term.
        state_rate = np.zeros(state.full_size)
    else:
        state_rate = np.asarray(
            dynamics_core(
                state,
                np.asarray(control, dtype=float),
                orbital_state,
                dmode=ErrorMode(
                    add_bias=False,
                    add_noise=False,
                    update_bias=False,
                    update_noise=False,
                ),
            ),
            dtype=float,
        )
        if state_rate.shape != (base,):
            raise ValueError(
                f"dynamics_core must return shape ({base},), got {state_rate.shape}"
            )

    source_mode = quaternion_mode if source_quaternion_mode is None else source_quaternion_mode
    target_mode = quaternion_mode if target_quaternion_mode is None else target_quaternion_mode
    source_map = state.tangent_map(
        quaternion_mode=source_mode, quaternion_order=quaternion_order
    )
    target_pinv = state.tangent_pinv(
        quaternion_mode=target_mode, quaternion_order=quaternion_order
    )
    target_rate = _tangent_map_rate(
        state,
        np.pad(state_rate, (0, state.full_size - state_rate.size)),
        quaternion_mode=target_mode,
        quaternion_order=quaternion_order,
    )
    source_to_target = target_pinv @ source_map
    return target_pinv @ (full @ source_map - target_rate @ source_to_target)


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
    *,
    final_state: EstimatorState | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and Van Loan-discretize the shared attitude process-noise model.

    The dynamics are integrated in ``source_quaternion_mode`` coordinates
    (falling back to ``quaternion_mode``), then mapped into
    ``target_quaternion_mode`` coordinates at ``final_state``.  This supports
    all source/target combinations between reduced attitude charts and the
    additive full-quaternion chart.
    """
    source_mode = kwargs.pop(
        "source_quaternion_mode",
        kwargs.get("quaternion_mode", State.DEFAULT_QUATERNION_MODE),
    )
    target_mode = kwargs.pop("target_quaternion_mode", source_mode)
    kwargs["quaternion_mode"] = source_mode

    transfer, continuous_psd = continuous_error_state_model(
        state, satellite, control, orbital_state, **kwargs
    )
    transition, discrete_psd = van_loan_discretize(transfer, continuous_psd, dt)

    if final_state is None:
        final_state = state
    if not isinstance(final_state, EstimatorState):
        raise TypeError(
            f"final_state must be an EstimatorState, got {type(final_state).__name__}"
        )
    if final_state.full_size != state.full_size:
        raise ValueError("state and final_state must have matching layouts")

    if source_mode == "full_quaternion":
        initial_projection = state.normalization_jacobian()
    else:
        initial_projection = np.eye(transition.shape[1])
    final_source_to_target = final_state.tangent_pinv(
        quaternion_mode=target_mode,
        quaternion_order=kwargs.get("quaternion_order", State.DEFAULT_QUATERNION_ORDER),
    ) @ final_state.tangent_map(
        quaternion_mode=source_mode,
        quaternion_order=kwargs.get("quaternion_order", State.DEFAULT_QUATERNION_ORDER),
    )

    transition = final_source_to_target @ transition @ initial_projection
    discrete_psd = final_source_to_target @ discrete_psd @ final_source_to_target.T

    return transition, discrete_psd
