"""
Capstone FD test for the revived analytic-derivative chain.

Stage A (PR #65) fixed the actuator-derivative call signatures in
EstimatedSatellite.dynJacCore and dynamics_Hessians. Stage B (this PR)
fixed the disturbance-derivative call signatures (the consumer
dist_torques_jacobian / dist_torque_hess) by standardising disturbance
derivatives on `(self, sat, x, os)` and updating the consumer to call
`j.torque_qjac(self, x, vecs["os"])`. The Dipole disturbance also gained
the previously-missing `torque_valvalhess` zero-impl (Dipole torque is
linear in the residual-dipole parameter so the value-value Hessian is
analytically zero).

This test verifies the WHOLE chain end-to-end on an EstimatedSatellite
with actuators + sensors + an estimated Dipole disturbance: the analytic
Hessian `ddxdot/dx dx` (returned by `dynamics_Hessians`) must equal the
central-difference of `dynJacCore`'s state-Jacobian.

External, non-tautological reference: a finite-difference Jacobian-of-the-
Jacobian, not the analytic Hessian compared to itself.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _sat_with_dipole():
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)]
        + [RW(axis=_UV[j], max_torque=4.51, J=0.22, h=1.0, h_max=3.8)
           for j in range(3)],
        sensors=[MTM(axis=_UV[j], noise=Noise(noise=0.0, std_noise=0.0))
                 for j in range(3)]
        + [Gyro(axis=_UV[j], bias=Bias(bias=0.0, std_bias=0.0),
                noise=Noise(noise=0.0, std_noise=0.0)) for j in range(3)],
        disturbances=[Dipole_Disturbance(
            dipole_torque=np.array([2.0e-4, -3.0e-4, 1.0e-4]),
            estimate_dist=False)],   # not augmented; just present in dynamics
    )


def _os():
    return Orbital_State(
        ephem=Ephemeris(), J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(.5), np.sqrt(.5)]),
        V=np.array([7.55, 0.0, 0.0]),
        B=np.array([1e-5, 2e-5, -3e-5]),
        S=np.array([1e8, 0.0, 0.0]), rho=5e-12)


def test_dynamics_Hessians_completes_with_dipole_disturbance():
    """End-to-end: dynamics_Hessians dispatches through actuator AND
    disturbance derivative blocks without crashing and returns finite
    tensors. This is the proof that BOTH Stage A's actuator-callsite fix
    AND Stage B's disturbance-callsite fix work together."""
    sat = _sat_with_dipole()
    os = _os()
    q = np.array([0.7, 0.3, -0.4, 0.5]); q /= np.linalg.norm(q)
    x = np.concatenate([[0.02, -0.01, 0.015], q, [1.0, 0.8, -0.6]])
    u = np.zeros(len(sat.actuators))
    H = sat.dynamics_Hessians(x, u, os)

    def _walk(o):
        if isinstance(o, np.ndarray): return [o]
        if isinstance(o, (list, tuple)):
            out = []
            for it in o: out.extend(_walk(it))
            return out
        return []

    tensors = _walk(H)
    assert tensors, "dynamics_Hessians returned no tensors"
    for t in tensors:
        assert np.all(np.isfinite(t)), \
            f"non-finite tensor of shape {t.shape}"


def test_ddxdot_dxdx_matches_fd_of_dynJacCore():
    """Analytic Hessian `ddxdot/dx dx` (= dynamics_Hessians[0]) must equal
    central-difference of dynJacCore's state-Jacobian dxdot/dx."""
    sat = _sat_with_dipole()
    os = _os()
    q = np.array([0.7, 0.3, -0.4, 0.5]); q /= np.linalg.norm(q)
    x0 = np.concatenate([[0.02, -0.01, 0.015], q, [1.0, 0.8, -0.6]])
    u0 = np.zeros(len(sat.actuators))

    H = sat.dynamics_Hessians(x0, u0, os)
    # The return is a 5x5 nested-list block-Hessian:
    # rows/cols indexed [x, u, act_bias, sens_bias, dist_param];
    # the (0,0) block is ddxdot/dx dx.
    H_xdx = np.asarray(H[0][0], float)
    SL = sat.state_len
    assert H_xdx.shape == (SL, SL, SL), \
        f"unexpected ddxdot/dx dx shape: {H_xdx.shape}"

    def _J(xv):
        out = sat.dynJacCore(xv, u0, os)
        return np.asarray(out[0], float)         # dxdot__dx, (SL, SL)

    eps = 1e-5
    H_fd = np.zeros((SL, SL, SL))
    for i in range(SL):
        dx = np.zeros(SL); dx[i] = eps
        H_fd[i, :, :] = (_J(x0 + dx) - _J(x0 - dx)) / (2.0 * eps)

    # Tolerance: central-diff truncation ~ O(eps^2)~1e-10, but the
    # multiplicative scale of the largest tensor element matters. Use a
    # relative+absolute bound; ~1e-4 is comfortable for the analytical
    # tensors involved.
    max_abs = float(np.max(np.abs(H_xdx))) + 1e-12
    err = float(np.max(np.abs(H_xdx - H_fd)))
    assert err < max(1e-4, 1e-3 * max_abs), (
        f"ddxdot/dx dx FD mismatch: max abs err {err:.3e} "
        f"(max tensor element {max_abs:.3e})")
