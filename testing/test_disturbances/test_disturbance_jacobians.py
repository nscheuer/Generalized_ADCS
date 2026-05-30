import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.disturbances import (
    Dipole_Disturbance,
    Drag_Disturbance,
    GG_Disturbance,
    GeometryConfig,
    GeometryFace,
    Prop_Disturbance,
    SRP_Disturbance,
)
from ADCS.satellite_hardware.satellite.satellite import Satellite


EPHEM = Ephemeris()
Q0 = normalize(np.array([1.0, 0.30, -0.20, 0.10]))
W0 = np.array([0.004, -0.003, 0.002])
X0 = np.concatenate([W0, Q0])


def make_orbital_state(**overrides) -> Orbital_State:
    values = dict(
        ephem=EPHEM,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=1e-4 * normalize(np.array([1.0, 2.0, -3.0])),
    )
    values.update(overrides)
    return Orbital_State(**values)


def call_disturbance_method(method, satellite, state, orbital_state):
    for kwargs in (
        dict(sat=satellite, x=state, os=orbital_state),
        dict(x=state, os=orbital_state),
        {},
    ):
        try:
            return method(**kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not call {method!r}")


def central_difference_quaternion_jacobian(torque_fn, quaternion, eps: float = 1e-7) -> np.ndarray:
    jacobian = np.zeros((4, 3))
    for index in range(4):
        delta = np.zeros(4)
        delta[index] = eps
        plus = np.asarray(torque_fn(np.concatenate([W0, quaternion + delta])), dtype=float).reshape(3)
        minus = np.asarray(torque_fn(np.concatenate([W0, quaternion - delta])), dtype=float).reshape(3)
        jacobian[index] = (plus - minus) / (2.0 * eps)
    return jacobian


def make_geometry_config(kind: str) -> GeometryConfig:
    if kind == "drag":
        faces = [
            GeometryFace(area=1.3, centroid=np.array([0.6, 0.1, 0.0]), normal=np.array([0.0, 1.0, 0.0]), CD=2.2),
            GeometryFace(
                area=0.7,
                centroid=np.array([-0.2, 0.0, 0.4]),
                normal=normalize(np.array([0.1, 1.0, 0.2])),
                CD=2.0,
            ),
        ]
    else:
        faces = [
            GeometryFace(
                area=1.3,
                centroid=np.array([0.6, 0.1, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                eta_a=0.05,
                eta_d=0.25,
                eta_s=0.70,
            ),
            GeometryFace(
                area=0.7,
                centroid=np.array([-0.2, 0.0, 0.4]),
                normal=normalize(np.array([1.0, 0.15, -0.1])),
                eta_a=0.10,
                eta_d=0.30,
                eta_s=0.55,
            ),
        ]
    return GeometryConfig(geometry_faces=faces)


def build_disturbance_case(kind: str):
    if kind == "gg":
        disturbance = GG_Disturbance()
        return disturbance, Satellite(J_0=np.diagflat([0.5, 0.8, 1.2]), disturbances=[disturbance]), make_orbital_state()
    if kind == "dipole":
        disturbance = Dipole_Disturbance(dipole_torque=np.array([0.4, -0.3, 0.2]))
        return disturbance, Satellite(disturbances=[disturbance]), make_orbital_state()
    if kind == "prop":
        disturbance = Prop_Disturbance(np.array([1e-4, -2e-4, 3e-4]))
        return disturbance, Satellite(disturbances=[disturbance]), make_orbital_state()
    if kind == "drag":
        disturbance = Drag_Disturbance(config=make_geometry_config("drag"))
        return disturbance, Satellite(disturbances=[disturbance]), make_orbital_state(rho=2.5e-12)
    if kind == "srp":
        disturbance = SRP_Disturbance(config=make_geometry_config("srp"))
        return disturbance, Satellite(disturbances=[disturbance]), make_orbital_state(S=np.array([1.5e8, 0.0, 0.0]))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["gg", "drag", "srp", "prop", "dipole"])
def test_disturbance_qjac_has_expected_shape(kind):
    disturbance, satellite, orbital_state = build_disturbance_case(kind)
    jacobian = np.asarray(call_disturbance_method(disturbance.torque_qjac, satellite, X0, orbital_state), dtype=float)
    assert jacobian.shape == (4, 3)


@pytest.mark.parametrize("kind", ["gg", "drag", "srp", "prop", "dipole"])
def test_disturbance_qjac_matches_central_difference(kind):
    disturbance, satellite, orbital_state = build_disturbance_case(kind)
    torque_fn = lambda state: call_disturbance_method(disturbance.torque, satellite, state, orbital_state)
    analytic = np.asarray(call_disturbance_method(disturbance.torque_qjac, satellite, X0, orbital_state), dtype=float)
    numeric = central_difference_quaternion_jacobian(torque_fn, Q0)
    scale = max(1.0, np.abs(numeric).max())
    assert np.abs(analytic - numeric).max() / scale < 1e-5


def test_gg_torque_qvac_matches_torque_qjac():
    disturbance, satellite, orbital_state = build_disturbance_case("gg")
    qvac = np.asarray(disturbance.torque_qvac(sat=satellite, x=X0, os=orbital_state), dtype=float)
    qjac = np.asarray(disturbance.torque_qjac(sat=satellite, x=X0, os=orbital_state), dtype=float)
    assert qvac.shape == (4, 3)
    assert np.allclose(qvac, qjac)


def test_srp_jacobian_is_zero_in_eclipse():
    disturbance, satellite, orbital_state = build_disturbance_case("srp")
    orbital_state._sunlit = False
    jacobian = np.asarray(disturbance.torque_qjac(sat=satellite, x=X0, os=orbital_state), dtype=float)
    assert jacobian.shape == (4, 3)
    assert np.allclose(jacobian, 0.0)


def test_srp_torque_is_zero_in_eclipse():
    disturbance, satellite, orbital_state = build_disturbance_case("srp")
    orbital_state._sunlit = False
    torque = np.asarray(disturbance.torque(sat=satellite, x=X0, os=orbital_state), dtype=float)
    assert np.allclose(torque, 0.0)


def test_prop_jacobian_is_callable_with_generic_signature():
    disturbance, satellite, orbital_state = build_disturbance_case("prop")
    jacobian = np.asarray(disturbance.torque_qjac(sat=satellite, x=X0, os=orbital_state), dtype=float)
    assert jacobian.shape == (4, 3)
    assert np.allclose(jacobian, 0.0)


def test_prop_hessian_has_expected_shape():
    disturbance, satellite, orbital_state = build_disturbance_case("prop")
    hessian = np.asarray(disturbance.torque_qqhess(sat=satellite, x=X0, os=orbital_state), dtype=float)
    assert hessian.shape == (4, 4, 3)
