from dataclasses import dataclass

import numpy as np

from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, MTM
from ADCS.state import State

_UV = MathConstants.unitvecs


@dataclass(frozen=True)
class HessianFDCase:
    satellite: EstimatedSatellite
    orbital_state: Orbital_State
    state: State
    control: np.ndarray


def _make_satellite() -> EstimatedSatellite:
    return EstimatedSatellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)]
        + [RW(axis=_UV[j], max_torque=4.51, J=0.22, h=1.0, h_max=3.8) for j in range(3)],
        sensors=[MTM(axis=_UV[j], noise=Noise(noise=0.0, std_noise=0.0)) for j in range(3)]
        + [
            Gyro(
                axis=_UV[j],
                bias=Bias(bias=0.0, std_bias=0.0),
                noise=Noise(noise=0.0, std_noise=0.0),
            )
            for j in range(3)
        ],
        disturbances=[
            Dipole_Disturbance(
                dipole_torque=np.array([2.0e-4, -3.0e-4, 1.0e-4]),
                estimate_dist=False,
            )
        ],
    )


def _make_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([7.55, 0.0, 0.0]),
        B=np.array([1e-5, 2e-5, -3e-5]),
        S=np.array([1e8, 0.0, 0.0]),
        rho=5e-12,
    )


def _make_case() -> HessianFDCase:
    satellite = _make_satellite()
    orbital_state = _make_orbital_state()
    quaternion = np.array([0.7, 0.3, -0.4, 0.5], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    state = State(w=[0.02, -0.01, 0.015], q=quaternion, h=[1.0, 0.8, -0.6])
    control = np.zeros(len(satellite.actuators))
    return HessianFDCase(
        satellite=satellite,
        orbital_state=orbital_state,
        state=state,
        control=control,
    )


def _walk_tensors(obj) -> list[np.ndarray]:
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(_walk_tensors(item))
        return tensors
    return []


def _state_hessian(case: HessianFDCase) -> np.ndarray:
    return np.asarray(
        case.satellite.dynamics_Hessians(case.state, case.control, case.orbital_state)[0][0],
        dtype=float,
    )


def _finite_difference_state_hessian(case: HessianFDCase, *, step: float = 1.0e-5) -> np.ndarray:
    state_len = case.satellite.state_len
    hessian = np.zeros((state_len, state_len, state_len))
    for index in range(state_len):
        delta = np.zeros(state_len)
        delta[index] = step
        jacobian_plus = np.asarray(
            case.satellite.dynJacCore(State.from_array(case.state.as_array() + delta), case.control, case.orbital_state)[0],
            dtype=float,
        )
        jacobian_minus = np.asarray(
            case.satellite.dynJacCore(State.from_array(case.state.as_array() - delta), case.control, case.orbital_state)[0],
            dtype=float,
        )
        hessian[index] = (jacobian_plus - jacobian_minus) / (2.0 * step)
    return hessian


def test_dynamics_hessians_returns_finite_tensors_with_disturbance() -> None:
    case = _make_case()

    tensors = _walk_tensors(
        case.satellite.dynamics_Hessians(case.state, case.control, case.orbital_state)
    )

    assert tensors, "dynamics_Hessians returned no tensors"
    for tensor in tensors:
        assert np.all(np.isfinite(tensor)), (
            f"dynamics_Hessians produced non-finite entries (shape {tensor.shape})"
        )


def test_state_hessian_matches_finite_difference_of_dynjaccore() -> None:
    case = _make_case()

    analytic = _state_hessian(case)
    numeric = _finite_difference_state_hessian(case)

    state_len = case.satellite.state_len
    assert analytic.shape == (state_len, state_len, state_len)

    max_abs = float(np.max(np.abs(analytic))) + 1.0e-12
    error = float(np.max(np.abs(analytic - numeric)))
    assert error < max(1.0e-4, 1.0e-3 * max_abs), (
        f"ddxdot/dxdx finite-difference mismatch: max abs err {error:.3e} "
        f"(max tensor element {max_abs:.3e})"
    )
