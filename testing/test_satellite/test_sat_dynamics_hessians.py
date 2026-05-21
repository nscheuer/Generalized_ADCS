"""
Finite-difference check for ``Satellite.dynamics_Hessians``.

This test compares the state Hessian returned by ``dynamics_Hessians`` with
central differences of ``dynJacCore`` and keeps the current non-functional
implementation visible via a strict ``xfail`` marker.
"""
import sys
import os
import numpy as np
import pytest

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris

@pytest.mark.slow
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
