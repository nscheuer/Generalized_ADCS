"""
RW command-Jacobian must respect actuator saturation (backlog #4).

After the saturation clamp (PR #34), RW torque = axis * clip(u, -u_max,
u_max) + bias + noise, so d(torque)/d(u) = axis strictly inside the limit
and 0 once saturated. The analytic Jacobians dtorq__du / dstor_torq__du
previously returned axis / -1 *unconditionally*, telling planners and
estimators the wheel still has full control authority in saturation.

These tests finite-difference the actual (noise-free) torque()/
storage_torque() and require the analytic Jacobian to match -- on BOTH
sides of the kink, with the FD step small enough never to straddle u_max.
RED before the fix (saturated region returned axis vs FD = 0).
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.actuators import RW
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris

EPHEM = Ephemeris()
UMAX = 0.05
X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _os():
    return Orbital_State(ephem=EPHEM, J2000=0.22,
                         R=np.array([7000.0, 0.0, 0.0]),
                         V=np.array([0.0, 7.5, 0.0]),
                         B=np.array([2e-5, -1e-5, 3e-5]))


def _rw():
    # no bias / no noise -> torque() is the deterministic clip map
    return RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=UMAX,
              J=0.01, h=0.0, h_max=0.2)


@pytest.mark.parametrize("u,saturated", [
    (0.4 * UMAX, False),     # strictly inside
    (-0.6 * UMAX, False),    # strictly inside, negative
    (2.5 * UMAX, True),      # strictly saturated
    (-3.0 * UMAX, True),     # strictly saturated, negative
])
def test_dtorq_du_matches_fd_across_saturation(u, saturated):
    rw, os_ = _rw(), _os()
    eps = 1e-7                                  # << distance to the kink
    fd = (np.asarray(rw.torque(u + eps, X, os_), float)
          - np.asarray(rw.torque(u - eps, X, os_), float)) / (2.0 * eps)
    ana = np.asarray(rw.dtorq__du(u, X, os_), float).reshape(3)
    assert np.allclose(ana, fd, atol=1e-6), f"u={u}: ana {ana} vs FD {fd}"
    if saturated:
        assert np.allclose(ana, 0.0), f"saturated: dtorq__du {ana} != 0"
    else:
        assert np.allclose(ana, rw.axis, atol=1e-9)


@pytest.mark.parametrize("u,saturated", [
    (0.3 * UMAX, False),
    (4.0 * UMAX, True),
])
def test_dstor_torq_du_matches_fd_across_saturation(u, saturated):
    rw, os_ = _rw(), _os()
    eps = 1e-7
    fd = (float(np.ravel(rw.storage_torque(u + eps, X, os_))[0])
          - float(np.ravel(rw.storage_torque(u - eps, X, os_))[0])) / (2 * eps)
    ana = float(np.ravel(rw.dstor_torq__du(u, X, os_))[0])
    assert np.isclose(ana, fd, atol=1e-6), f"u={u}: ana {ana} vs FD {fd}"
    assert np.isclose(ana, 0.0 if saturated else -1.0, atol=1e-9)
