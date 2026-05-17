"""
Tracking guard for Satellite.dynamics_Hessians (backlog #3).

dynamics_Hessians is staged scaffolding for a one-step look-ahead MPC
controller that is NOT yet (re)implemented and is called by nothing. It had
`J = self.J` (AttributeError; fixed here to self.J_COM, consistent with the
PR #33 gyroscopic-inertia convention). A second, deeper defect remains: the
actuator second-derivative calls (dtorq__dbasestate, ddstor_torq__*) are
invoked with a stale signature (an extra `self` arg) — older actuator API,
never updated. Fully resurrecting + FD-verifying a 165-line second-order
tensor belongs WITH the MPC re-implementation (its only consumer), not in
isolation.

This test FD-verifies the Hessian against central differences of
dynJacCore and is marked strict xfail: it currently fails (the method
raises), so its broken/pending state is *tracked* rather than silently
green. When the MPC re-implementation makes dynamics_Hessians functional
AND finite-difference-correct, this XPASSes — and because strict=True, CI
then FAILS, signalling "remove this xfail / the MPC scaffolding is live".
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris


@pytest.mark.xfail(strict=True, reason=(
    "dynamics_Hessians is non-functional scaffolding for the not-yet-"
    "reimplemented one-step look-ahead MPC: actuator 2nd-derivative calls "
    "use a stale (extra-self) signature. XPASS here = the MPC work has made "
    "it functional + FD-correct; remove this xfail when that happens."
))
def test_dynamics_hessians_matches_finite_difference_of_jacobian():
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=np.array([7000.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.zeros(3), S=np.array([1e8, 0.0, 0.0]), rho=0.0)
    mtq = MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)
    sat = Satellite(mass=4.0, J_0=np.diagflat([0.5, 0.8, 1.2]),
                    actuators=[mtq])
    x = np.hstack(([0.02, -0.01, 0.015], [1.0, 0.0, 0.0, 0.0]))
    u = np.zeros(1)

    # The Hessian of the dynamics w.r.t. state == d(dynJacCore.dxdot__dx)/dx.
    H = sat.dynamics_Hessians(x, u, os0)           # currently raises -> xfail
    ddxdot__dxdx = np.asarray(H[0][0], dtype=float)

    eps = 1e-6
    n = x.size
    fd = np.zeros((n, n, ddxdot__dxdx.shape[-1]))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        Jp = np.asarray(sat.dynJacCore(x + dx, u, os0)[0], dtype=float)
        Jm = np.asarray(sat.dynJacCore(x - dx, u, os0)[0], dtype=float)
        fd[i] = (Jp - Jm) / (2.0 * eps)

    err = np.max(np.abs(ddxdot__dxdx[:n, :n, :] - fd))
    assert err < 1e-4, f"dynamics_Hessians vs FD(dynJacCore): max err {err:.2e}"
