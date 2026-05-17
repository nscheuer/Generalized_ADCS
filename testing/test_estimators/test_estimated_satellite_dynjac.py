"""
EstimatedSatellite.dynJacCore was dead-on-arrival: `q = x[4:7]` (a 3-element
slice; the quaternion is x[3:7] everywhere else) crashed rot_mat(q) on any
normal state. It is the augmented analytic Jacobian (intended for an EKF /
the to-be-reimplemented MPC) and is never called by the sigma-point SRUKF,
so the bug went untested.

This FD guard exercises it on a reaction-wheel config (no disturbances --
the disturbance-Jacobian path has a separate stale-signature bug,
dist_torques_jacobian -> torque_qjac(self,vecs), deferred to land with the
#35 disturbance-Jacobian fixes). RED before the fix (rot_mat ValueError),
GREEN after (dxdot/dx matches central FD of dynamics_core to ~1e-6, same
methodology PR #33 used to verify the base dynJacCore).
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_constants import MathConstants


@pytest.mark.xfail(strict=True, reason=(
    "EstimatedSatellite.dynJacCore is pervasively API-rotted dead "
    "scaffolding for the not-yet-reimplemented EKF/MPC (same family as "
    "dynamics_Hessians #3/#44): beyond the q=x[4:7] slice (fixed here), it "
    "calls actuator dtorq__dbasestate / ddstor_torq__* with a stale extra-"
    "self signature and dist_torques_jacobian with a stale torque_qjac "
    "signature. The q fix is real & shipped; full resurrection + FD "
    "verification belongs WITH the EKF/MPC reimplementation. XPASS here "
    "(strict -> CI fails) signals that work is done -> remove this xfail."))
def test_estimated_satellite_dynjac_matches_fd_with_rw():
    ep = Ephemeris()
    os0 = Orbital_State(ephem=ep, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([2e-5, -1e-5, 3e-5]),
                        S=np.array([1e8, 0.0, 0.0]), rho=0.0)
    rws = [RW(axis=j, max_torque=4.51, J=0.22, h=1.0, h_max=3.8)
           for j in MathConstants.unitvecs]
    mtq = MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)
    est = EstimatedSatellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                             actuators=[mtq] + rws)        # no disturbances
    SL = est.state_len
    q = np.array([0.7, 0.3, -0.4, 0.5]); q = q / np.linalg.norm(q)
    x = np.concatenate([[0.02, -0.01, 0.015], q, [1.0, 0.8, -0.6]])
    u = np.zeros(len(est.actuators))

    J = np.asarray(est.dynJacCore(x, u, os0)[0], float)   # (SL,SL), layout [in,out]
    assert J.shape == (SL, SL)

    eps = 1e-7
    fd = np.zeros((SL, SL))                                # [out,in]
    for i in range(SL):
        dx = np.zeros(SL); dx[i] = eps
        fp = np.asarray(est.dynamics_core(x + dx, u, os0), float)
        fm = np.asarray(est.dynamics_core(x - dx, u, os0), float)
        fd[:, i] = (fp - fm) / (2.0 * eps)

    err = np.max(np.abs(J.T - fd))
    assert err < 1e-5, f"EstimatedSatellite.dynJacCore vs FD: max err {err:.2e}"
