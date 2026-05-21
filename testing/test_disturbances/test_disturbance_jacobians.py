"""
Regression tests for disturbance quaternion Jacobians.

These tests compare each disturbance model's analytic ``torque_qjac`` against a
central finite-difference derivative of ``torque()`` with respect to the raw
quaternion, using representative operating points chosen to avoid
non-differentiable contact or illumination boundaries. The module also checks a
few disturbance-specific invariants, including Jacobian shape, eclipse
behaviour for SRP, and generic call compatibility for gravity-gradient and
propulsion disturbance helpers.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.disturbances import (
    SRP_Disturbance, Drag_Disturbance, Prop_Disturbance,
    Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace,
)
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize

EPHEM = Ephemeris()

# Fixed, non-identity attitude (raw unit quaternion, scalar-first).
Q0 = normalize(np.array([1.0, 0.30, -0.20, 0.10]))
W0 = np.array([0.004, -0.003, 0.002])
X0 = np.concatenate([W0, Q0])


def _os(**kw):
    base = dict(ephem=EPHEM, J2000=0.22,
                R=np.array([7000.0, 0.0, 0.0]),
                V=np.array([0.0, 8.0, 0.0]),
                B=1e-4 * normalize(np.array([1.0, 2.0, -3.0])))
    base.update(kw)
    return Orbital_State(**base)


def _call(method, sat, x, os_):
    """Disturbance methods use either (sat, x, os) or (x, os) or ()."""
    for args, kwargs in (
        ((), dict(sat=sat, x=x, os=os_)),
        ((), dict(x=x, os=os_)),
        ((), {}),
    ):
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not call {method!r}")


def _fd_jac(torque_fn, q, eps=1e-7):
    """Central FD of torque wrt the RAW quaternion -> (4,3) = [q, tau]."""
    fd = np.zeros((4, 3))
    for j in range(4):
        dq = np.zeros(4)
        dq[j] = eps
        tp = np.asarray(torque_fn(np.concatenate([W0, q + dq])), float).reshape(3)
        tm = np.asarray(torque_fn(np.concatenate([W0, q - dq])), float).reshape(3)
        fd[j] = (tp - tm) / (2.0 * eps)
    return fd


def _drag_srp_cfg(kind):
    # Two faces, both strictly lit at Q0, with non-zero lever arms so the
    # multi-face per-face mask path is exercised (the old bug's blast radius).
    rmat_eci2b = None  # filled per call; faces chosen body-frame-aligned below
    if kind == "drag":
        faces = [
            GeometryFace(area=1.3, centroid=np.array([0.6, 0.1, 0.0]),
                         normal=np.array([0.0, 1.0, 0.0]), CD=2.2),
            GeometryFace(area=0.7, centroid=np.array([-0.2, 0.0, 0.4]),
                         normal=normalize(np.array([0.1, 1.0, 0.2])), CD=2.0),
        ]
    else:
        faces = [
            GeometryFace(area=1.3, centroid=np.array([0.6, 0.1, 0.0]),
                         normal=np.array([1.0, 0.0, 0.0]),
                         eta_a=0.05, eta_d=0.25, eta_s=0.70),
            GeometryFace(area=0.7, centroid=np.array([-0.2, 0.0, 0.4]),
                         normal=normalize(np.array([1.0, 0.15, -0.1])),
                         eta_a=0.10, eta_d=0.30, eta_s=0.55),
        ]
    return GeometryConfig(geometry_faces=faces)


def _build(kind):
    if kind == "gg":
        d = GG_Disturbance()
        sat = Satellite(J_0=np.diagflat([0.5, 0.8, 1.2]), disturbances=[d])
        return d, sat, _os()
    if kind == "dipole":
        d = Dipole_Disturbance(dipole_torque=np.array([0.4, -0.3, 0.2]))
        return d, Satellite(disturbances=[d]), _os()
    if kind == "prop":
        d = Prop_Disturbance(np.array([1e-4, -2e-4, 3e-4]))
        return d, Satellite(disturbances=[d]), _os()
    if kind == "drag":
        d = Drag_Disturbance(config=_drag_srp_cfg("drag"))
        return d, Satellite(disturbances=[d]), _os(rho=2.5e-12)
    if kind == "srp":
        d = SRP_Disturbance(config=_drag_srp_cfg("srp"))
        # Sun far along +x (ECI) -> body-frame sun ~ +x, faces above are lit.
        return d, Satellite(disturbances=[d]), _os(S=np.array([1.5e8, 0.0, 0.0]))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["gg", "drag", "srp", "prop", "dipole"])
def test_disturbance_qjac_matches_central_fd(kind):
    d, sat, os_ = _build(kind)

    torque_fn = lambda x: _call(d.torque, sat, x, os_)
    jac = np.asarray(_call(d.torque_qjac, sat, X0, os_), float)

    assert jac.shape == (4, 3), f"{kind}: torque_qjac shape {jac.shape}, want (4,3)"

    fd = _fd_jac(torque_fn, Q0)
    scale = max(1.0, np.abs(fd).max())
    err = np.abs(jac - fd).max() / scale
    assert err < 1e-5, (
        f"{kind}: analytic torque_qjac vs central FD mismatch, "
        f"max rel err {err:.2e}\nanalytic=\n{jac}\nFD=\n{fd}"
    )


def test_gg_torque_qvac_no_longer_raises():
    d, sat, os_ = _build("gg")
    j = np.asarray(d.torque_qvac(sat=sat, x=X0, os=os_), float)
    assert j.shape == (4, 3) and np.isfinite(j).all()
    # torque_qjac is an alias of torque_qvac.
    assert np.allclose(j, np.asarray(d.torque_qjac(sat=sat, x=X0, os=os_), float))


def test_srp_jacobian_zero_and_correct_shape_in_eclipse():
    d, sat, os_ = _build("srp")
    os_._sunlit = False  # force eclipse (Orbital_State honours _sunlit)
    assert not os_.is_sunlit()
    j = np.asarray(d.torque_qjac(sat=sat, x=X0, os=os_), float)
    assert j.shape == (4, 3), f"eclipse SRP jac shape {j.shape}, want (4,3)"
    assert np.allclose(j, 0.0)
    # torque itself is zero in eclipse, so the (smooth) FD is also zero.
    assert np.allclose(np.asarray(d.torque(sat=sat, x=X0, os=os_), float), 0.0)


def test_prop_jacobian_callable_generically_and_zero():
    d, sat, os_ = _build("prop")
    # Generic (sat, x, os) call must not raise (old no-arg signature did).
    j = np.asarray(d.torque_qjac(sat=sat, x=X0, os=os_), float)
    assert j.shape == (4, 3) and np.allclose(j, 0.0)
    assert np.asarray(d.torque_qqhess(sat=sat, x=X0, os=os_), float).shape == (4, 4, 3)
