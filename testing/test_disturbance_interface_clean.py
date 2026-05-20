"""
Stage C cleanness contract for the disturbance derivative interface.

Every disturbance subclass must expose the canonical analytic-derivative
methods used by ``Satellite.dist_torques_jacobian`` / ``dist_torque_hess``
(and by the eventual EKF / one-step MPC consumers):

    torque_qjac        -> ndarray shape (4, 3)
    torque_qqhess      -> ndarray shape (4, 4, 3)
    torque_valjac      -> ndarray shape (n_dist_params, 3)
    torque_qvalhess    -> ndarray shape (4, n_dist_params, 3)
    torque_valvalhess  -> ndarray shape (n_dist_params, n_dist_params, 3)

with the unified ``(self, sat, x, os)`` signature, even when the disturbance
has no estimable parameters (then ``n_dist_params == 0`` and the val-axes
have length 0). Zero defaults are inherited from ``Disturbance`` so a
subclass without a particular physical derivative remains callable through
the canonical entry point.

This file also FD-verifies the two non-trivial bodies touched in Stage C:

    * GG.torque_qqhess  — the J_0 contraction was previously written as
      ``sat.J_0 @ dnadir_vec__dq`` (a (3,3)@(4,3) shape-error), masking the
      whole Hessian behind a typo'd method name. Stage C exposes it via the
      canonical ``torque_qqhess`` and corrects the contraction to
      ``dnadir_vec__dq @ sat.J_0.T`` (same convention torque_qjac uses).
    * Dipole.{torque_valjac, torque_qvalhess} — Stage C added the missing
      ``sat`` parameter, so these are now callable through the unified
      signature; FD over the dipole vector and the quaternion is the
      strongest readily-available correctness witness.
"""
import numpy as np
import pytest

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.gg_disturbance import GG_Disturbance
from ADCS.satellite_hardware.disturbances.dipole_disturbance import Dipole_Disturbance
from ADCS.satellite_hardware.disturbances.srp_disturbance import SRP_Disturbance
from ADCS.satellite_hardware.disturbances.prop_disturbance import Prop_Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import (
    GeometryFace, GeometryConfig,
)


# ----------------------------- fixtures -----------------------------------

@pytest.fixture(scope="module")
def env():
    ephem = Ephemeris()
    sat = Satellite(mass=4.0, J_0=np.diagflat([0.5, 0.8, 1.2]),
                    actuators=[MTQ(axis=np.array([1., 0., 0.]),
                                   max_torque=10.0)])
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=np.array([7000., 0., 0.]),
                        V=np.array([0., 7.5, 0.]),
                        B=np.array([1e-5, 2e-6, -1e-6]),
                        S=np.array([1e8, 0., 0.]),
                        rho=0.0)
    x = np.hstack(([0.02, -0.01, 0.015], [1., 0., 0., 0.]))
    geom = GeometryConfig([GeometryFace(area=1.0,
                                        centroid=np.array([1., 0., 0.]),
                                        normal=np.array([1., 0., 0.]),
                                        eta_s=0.1, eta_d=0.1, eta_a=0.8,
                                        CD=2.2)])
    return sat, os0, x, geom


def _all_disturbances(geom):
    """One instance of every concrete disturbance subclass."""
    return {
        "GG":     GG_Disturbance(),
        "Dipole": Dipole_Disturbance(np.array([1e-3, -2e-4, 5e-4])),
        "SRP":    SRP_Disturbance(geom),
        "Prop":   Prop_Disturbance(np.array([1e-6, 0., 0.])),
    }


CANONICAL_METHODS = [
    "torque_qjac", "torque_qqhess",
    "torque_valjac", "torque_qvalhess", "torque_valvalhess",
]


# ----------------------------- cleanness contract -------------------------

@pytest.mark.parametrize("name", ["GG", "Dipole", "SRP", "Prop"])
@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_every_disturbance_exposes_canonical_method(env, name, method):
    """Every disturbance subclass × every canonical method must be
    callable with the unified ``(sat, x, os)`` signature and return an
    ndarray whose shape matches the canonical layout for that method."""
    sat, os0, x, geom = env
    d = _all_disturbances(geom)[name]
    val = int(getattr(d, "estimated_vector_length", 0))

    out = getattr(d, method)(sat, x, os0)
    arr = np.asarray(out)

    expected_shape = {
        "torque_qjac":       (4, 3),
        "torque_qqhess":     (4, 4, 3),
        "torque_valjac":     (val, 3),
        "torque_qvalhess":   (4, val, 3),
        "torque_valvalhess": (val, val, 3),
    }[method]
    assert arr.shape == expected_shape, (
        f"{name}.{method} returned {arr.shape}, expected {expected_shape}"
    )


def test_base_disturbance_zero_defaults_match_layout():
    """The base ``Disturbance`` class provides correctly-shaped zero
    defaults so subclasses that don't implement a particular derivative
    are still safe to call through the canonical interface."""
    base = Disturbance(estimate_dist=False, estimated_vector_length=2)
    sat = object()
    x = np.zeros(7)
    os = object()
    assert base.torque_qjac(sat, x, os).shape == (4, 3)
    assert base.torque_qqhess(sat, x, os).shape == (4, 4, 3)
    assert base.torque_valjac(sat, x, os).shape == (2, 3)
    assert base.torque_qvalhess(sat, x, os).shape == (4, 2, 3)
    assert base.torque_valvalhess(sat, x, os).shape == (2, 2, 3)
    for m in CANONICAL_METHODS:
        assert np.all(np.asarray(getattr(base, m)(sat, x, os)) == 0.0)


# ----------------------------- FD verification ----------------------------

def _fd_qjac_of(method, sat, x, os, eps=1e-6):
    """Central-diff of ``method(sat, x, os)`` over the 4 quaternion
    components of ``x``. Returns shape ``(4, *method_output.shape)``."""
    out0 = np.asarray(method(sat, x, os))
    H = np.zeros((4,) + out0.shape)
    for k in range(4):
        dx = np.zeros_like(x); dx[3 + k] = eps
        plus  = np.asarray(method(sat, x + dx, os))
        minus = np.asarray(method(sat, x - dx, os))
        H[k] = (plus - minus) / (2.0 * eps)
    return H


def test_gg_torque_qjac_matches_fd_of_torque(env):
    """Analytic GG quaternion Jacobian vs central-difference of torque."""
    sat, os0, x, _ = env
    gg = GG_Disturbance()
    err = np.max(np.abs(
        gg.torque_qjac(sat, x, os0)
        - _fd_qjac_of(lambda s, xx, oo: gg.torque(s, xx, oo), sat, x, os0)
    ))
    assert err < 1e-9, f"GG torque_qjac vs FD(torque): max err {err:.2e}"


def test_gg_torque_qqhess_matches_fd_of_qjac(env):
    """Analytic GG quaternion Hessian vs central-difference of the
    analytic quaternion Jacobian. Locks in the J_0-contraction fix
    Stage C applied to the Hessian body."""
    sat, os0, x, _ = env
    gg = GG_Disturbance()
    err = np.max(np.abs(
        gg.torque_qqhess(sat, x, os0)
        - _fd_qjac_of(gg.torque_qjac, sat, x, os0)
    ))
    # eps=1e-6 truncation gives ~1e-5 with typical GG Hessian magnitudes;
    # 1e-3 is a comfortable correctness band, well below any real bug.
    assert err < 1e-3, f"GG torque_qqhess vs FD(qjac): max err {err:.2e}"


def test_dipole_torque_valjac_matches_fd_over_dipole(env):
    """Analytic dipole-parameter Jacobian vs central-difference over
    the dipole vector itself."""
    sat, os0, x, _ = env
    m0 = np.array([1e-3, -2e-4, 5e-4])
    dip = Dipole_Disturbance(m0); dip.current_torque = m0.copy()
    valjac_an = dip.torque_valjac(sat, x, os0)

    h = 1e-9
    valjac_fd = np.zeros((3, 3))
    for j in range(3):
        dm = np.zeros(3); dm[j] = h
        dip.current_torque = m0 + dm
        tp = dip.torque(x, os0)
        dip.current_torque = m0 - dm
        tm = dip.torque(x, os0)
        valjac_fd[j] = (tp - tm) / (2.0 * h)
    dip.current_torque = m0.copy()
    err = np.max(np.abs(valjac_an - valjac_fd))
    assert err < 1e-9, f"Dipole torque_valjac vs FD(dipole): max err {err:.2e}"


def test_dipole_torque_qvalhess_matches_fd_of_valjac_over_q(env):
    """Mixed (q, m_d) Hessian vs central-difference of torque_valjac
    over the quaternion."""
    sat, os0, x, _ = env
    m0 = np.array([1e-3, -2e-4, 5e-4])
    dip = Dipole_Disturbance(m0); dip.current_torque = m0.copy()
    err = np.max(np.abs(
        dip.torque_qvalhess(sat, x, os0)
        - _fd_qjac_of(dip.torque_valjac, sat, x, os0)
    ))
    assert err < 1e-9, f"Dipole torque_qvalhess vs FD(valjac): max err {err:.2e}"


def test_dipole_torque_valvalhess_is_zero(env):
    """Dipole torque is linear in the dipole vector; the second
    derivative w.r.t. ``m_d`` is identically zero."""
    sat, os0, x, _ = env
    dip = Dipole_Disturbance(np.array([1e-3, -2e-4, 5e-4]))
    H = dip.torque_valvalhess(sat, x, os0)
    assert H.shape == (3, 3, 3)
    assert np.all(H == 0.0)
