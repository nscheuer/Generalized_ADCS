"""
Stage A of the analytic-derivative revival: fix the stale actuator call
signatures in dynamics_Hessians / dynJacCore (PR #44 and PR #47 each
documented this as a deferred defect).

The 30+ call sites across Satellite.dynamics_Hessians,
EstimatedSatellite.dynamics_Hessians and EstimatedSatellite.dynJacCore
invoked actuator derivative methods as `(u[j], self, x, vecs)` (extra
`self`, a dict in place of orbital_state), while every actuator method --
including the base zero-implementations -- has signature
`(self, u, x, os)`. Result: TypeError on the very first actuator call,
killing dynamics_Hessians / dynJacCore at the actuator block before
reaching the disturbance block.

This test exercises dynamics_Hessians end-to-end on a satellite with
actuators only and no disturbances -- so the (still-rotted, Stage B)
disturbance block iterates an empty list and dynamics_Hessians completes.
RED on origin/main (TypeError at the first actuator call), GREEN after.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _sat_no_disturbances():
    # EstimatedSatellite (the override is the live consumer of these
    # derivatives -- one-step MPC / augmented EKF). No disturbances ->
    # dist_param_inds is empty, the disturbance block in dynamics_Hessians
    # iterates an empty list, and the (still-rotted, Stage B) disturbance
    # call signatures are not exercised. So Stage A's actuator-signature
    # fix alone makes dynamics_Hessians complete end-to-end here.
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)]
        + [RW(axis=_UV[j], max_torque=4.51, J=0.22, h=1.0, h_max=3.8)
           for j in range(3)],
        sensors=[],
    )


def _os():
    return Orbital_State(
        ephem=Ephemeris(), J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(.5), np.sqrt(.5)]),
        V=np.array([7.55, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1, 0.0, 0.0]), rho=5e-12)


def test_dynamics_Hessians_actuator_block_now_dispatches_correctly():
    """End-to-end dynamics_Hessians on a satellite with actuators only
    must complete (no disturbances -> disturbance block is empty) and
    return tensors of the expected shape and finite values.

    RED on origin/main: TypeError on the very first actuator call
    (`MTQ.dtorq__dbasestate() takes 4 positional arguments but 5 were given`)
    inside the actuator block, before the disturbance block.
    GREEN after the Stage A fix.
    """
    sat = _sat_no_disturbances()
    os = _os()
    q = np.array([0.7, 0.3, -0.4, 0.5]); q /= np.linalg.norm(q)
    x = np.concatenate([[0.02, -0.01, 0.015], q, [1.0, 0.8, -0.6]])
    u = np.zeros(len(sat.actuators))

    H = sat.dynamics_Hessians(x, u, os)
    # The exact return container is List[List[ndarray]] per the docstring;
    # the structural guarantees we care about are:
    # (a) it doesn't raise, (b) every tensor is finite.
    def _walk(o):
        if isinstance(o, np.ndarray):
            return [o]
        if isinstance(o, (list, tuple)):
            out = []
            for it in o:
                out.extend(_walk(it))
            return out
        return []

    tensors = _walk(H)
    assert tensors, "dynamics_Hessians returned no tensors"
    for t in tensors:
        assert np.all(np.isfinite(t)), \
            f"dynamics_Hessians produced non-finite entries (shape {t.shape})"
