<<<<<<< HEAD
import numpy as np
import pytest

from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite


@pytest.fixture(scope="module")
def orbital_state():
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.zeros(3),
        S=np.array([1e8, 0.0, 0.0]),
        rho=0.0,
    )


def body_inertia() -> np.ndarray:
    return np.diagflat([0.5, 0.8, 1.2])


def inertial_angular_momentum(satellite: Satellite, state: np.ndarray, axes: np.ndarray | None = None) -> np.ndarray:
    body_momentum = satellite.J_COM @ state[:3]
    if axes is not None and state.size > 7:
        body_momentum = body_momentum + axes.T @ state[7:]
    return rot_mat(state[3:7]) @ body_momentum


def rotational_kinetic_energy(satellite: Satellite, state: np.ndarray) -> float:
    return 0.5 * state[:3] @ (satellite.J_COM @ state[:3])


=======
"""Regression tests for rigid-body attitude dynamics correctness.

These capture two bugs the existing suite missed:

1. ``dynamics_core`` / ``dynJacCore`` used ``self.J_0`` (inertia about the
   reference origin) in the gyroscopic ``w x (J w)`` term while inverting
   ``invJ_noRW`` (derived from the center-of-mass inertia).  With a non-zero
   ``COM`` offset these are different tensors, so torque-free motion did not
   conserve angular momentum.  Every pre-existing dynamics test uses
   ``COM=0`` (where ``J_COM == J_0``), so the bug was invisible.

2. ``noiseless_rk4`` renormalised ``x[3:7]`` in place, mutating the caller's
   input array.

The COM=0 path is mathematically a no-op under the fix (``J_COM == J_0``),
so the full pre-existing suite must remain green.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import rot_mat


@pytest.fixture(scope="module")
def os0():
    # No actuators / no disturbances are used below, so B, S, rho are inert;
    # the run is genuinely torque-free.
    ephem = Ephemeris()
    return Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
        B=np.zeros(3), S=np.array([1e8, 0.0, 0.0]), rho=0.0,
    )


def _Jbody():
    return np.diagflat([0.5, 0.8, 1.2])


def _Heci(sat, x, A=None):
    """True inertial angular momentum about the COM."""
    Hb = sat.J_COM @ x[:3]
    if A is not None and x.size > 7:
        Hb = Hb + A.T @ x[7:]
    return rot_mat(x[3:7]) @ Hb


def _rot_ke(sat, x):
    w = x[:3]
    return 0.5 * w @ (sat.J_COM @ w)


# Swept COM offsets: small, single-axis (x, y), multi-axis, and a near-zero
# degenerate case where J_COM ~= J_0 (guards that the fix stays a no-op there).
# mass=2.0 keeps J_COM positive-definite for every offset below.
>>>>>>> 3dd3be9 (moved tests and updated estimatedsatellite)
COM_OFFSETS = [
    np.array([0.05, 0.02, -0.03]),
    np.array([0.20, 0.0, 0.0]),
    np.array([0.0, -0.15, 0.0]),
    np.array([0.10, 0.10, -0.10]),
    np.array([1e-6, 0.0, 0.0]),
]
<<<<<<< HEAD
COM_IDS = ["small", "x_axis", "y_axis", "multi", "near_zero"]


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_torque_free_motion_conserves_angular_momentum_with_com_offset(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    state = np.hstack(([0.02, -0.015, 0.01], [1.0, 0.0, 0.0, 0.0]))
    initial_momentum = inertial_angular_momentum(satellite, state)

    max_drift = 0.0
    for _ in range(4000):
        state = satellite.noiseless_rk4(state, np.zeros(0), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(
            max_drift,
            np.linalg.norm(inertial_angular_momentum(satellite, state) - initial_momentum) / np.linalg.norm(initial_momentum),
        )

    assert max_drift < 1e-6


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_torque_free_motion_conserves_rotational_energy_with_com_offset(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    state = np.hstack(([0.02, -0.015, 0.01], [1.0, 0.0, 0.0, 0.0]))
    initial_energy = rotational_kinetic_energy(satellite, state)

    max_drift = 0.0
    for _ in range(4000):
        state = satellite.noiseless_rk4(state, np.zeros(0), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(max_drift, abs(rotational_kinetic_energy(satellite, state) - initial_energy) / abs(initial_energy))

    assert max_drift < 1e-6


def test_torque_free_motion_with_rw_conserves_total_angular_momentum(orbital_state):
    from ADCS.helpers.math_helpers import normalize
    from ADCS.satellite_hardware.actuators import RW

    wheel = RW(axis=normalize(np.array([1.0, 0.6, 0.3])), max_torque=1.0, J=0.05, h=0.02, h_max=10.0)
    satellite = Satellite(mass=3.0, COM=np.array([0.04, -0.02, 0.05]), J_0=body_inertia(), actuators=[wheel])
    axes = np.vstack([actuator.axis for actuator in satellite.rw_actuators])
    state = np.hstack(([0.02, -0.015, 0.01], [1.0, 0.0, 0.0, 0.0], [0.02]))
    initial_momentum = inertial_angular_momentum(satellite, state, axes)

    max_drift = 0.0
    for _ in range(3000):
        state = satellite.noiseless_rk4(state, np.zeros(1), 0.1, orbital_state, orbital_state, mid_orbital_state=orbital_state)
        max_drift = max(
            max_drift,
            np.linalg.norm(inertial_angular_momentum(satellite, state, axes) - initial_momentum) / np.linalg.norm(initial_momentum),
        )

    assert max_drift < 1e-5


@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_dynamics_core_matches_analytic_euler_equation(orbital_state, com):
    satellite = Satellite(mass=2.0, COM=com, J_0=body_inertia())
    rng = np.random.default_rng(0)
    for _ in range(20):
        omega = rng.normal(size=3) * 0.05
        state = np.hstack((omega, [1.0, 0.0, 0.0, 0.0]))
        state_dot = satellite.dynamics_core(x=state, u=np.zeros(0), orbital_state=orbital_state)
        analytic = satellite.invJ_COM @ (-np.cross(omega, satellite.J_COM @ omega))
        assert np.allclose(state_dot[:3], analytic, atol=1e-10)


def test_dynjac_matches_finite_difference_with_com_offset(orbital_state):
    from ADCS.satellite_hardware.actuators import MTQ

    satellite = Satellite(
        mass=4.0,
        COM=np.array([0.05, 0.02, -0.03]),
        J_0=body_inertia(),
        actuators=[MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)],
    )
    state = np.hstack(([0.03, -0.02, 0.012], [1.0, 0.0, 0.0, 0.0]))
    control = np.zeros(1)

    def dynamics(candidate):
        return satellite.dynamics_core(x=candidate, u=control, orbital_state=orbital_state)

    jacobian = satellite.dynJacCore(state, control, orbital_state)[0]
    numeric = np.zeros((state.size, state.size))
    for index in range(state.size):
        delta = np.zeros(state.size)
        delta[index] = 1e-7
        numeric[:, index] = (dynamics(state + delta) - dynamics(state - delta)) / (2e-7)

    assert np.allclose(jacobian.T[: state.size, : state.size], numeric, atol=1e-4)


def test_noiseless_rk4_does_not_mutate_input_state(orbital_state):
    satellite = Satellite(J_0=body_inertia())
    state = np.array([0.01, -0.008, 0.006, 2.0, 0.0, 0.0, 0.0])
    before = state.copy()
    satellite.noiseless_rk4(state, np.zeros(0), 0.5, orbital_state, orbital_state, mid_orbital_state=orbital_state)
    assert np.array_equal(state, before)
=======
COM_IDS = ["small", "x-axis", "y-axis", "multi", "near-zero"]


# --------------------------------------------------------------------------
# Bug 1a: torque-free angular-momentum / energy conservation with a COM offset
# --------------------------------------------------------------------------
@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_torque_free_conservation_with_com_offset(os0, com):
    sat = Satellite(mass=2.0, COM=com, J_0=_Jbody())

    x = np.hstack(([0.02, -0.015, 0.01], [1.0, 0.0, 0.0, 0.0]))
    H0 = _Heci(sat, x)
    E0 = _rot_ke(sat, x)

    dt, N = 0.1, 4000
    maxdH = maxdE = 0.0
    for _ in range(N):
        x = sat.noiseless_rk4(x, np.zeros(0), dt, os0, os0, mid_orbital_state=os0)
        maxdH = max(maxdH, np.linalg.norm(_Heci(sat, x) - H0) / np.linalg.norm(H0))
        maxdE = max(maxdE, abs(_rot_ke(sat, x) - E0) / abs(E0))

    # RK4 truncation only; the J_0/J_COM bug produced ~1.3e-2 here.
    assert maxdH < 1e-6, f"angular momentum drift {maxdH:.2e}"
    assert maxdE < 1e-6, f"rotational KE drift {maxdE:.2e}"


# --------------------------------------------------------------------------
# Bug 1b: same, with reaction wheels present (J_0 != J_COM != J_noRW)
# --------------------------------------------------------------------------
def test_torque_free_conservation_com_offset_with_rw(os0):
    from ADCS.satellite_hardware.actuators import RW
    from ADCS.helpers.math_helpers import normalize

    ax = normalize(np.array([1.0, 0.6, 0.3]))
    rw = RW(axis=ax, max_torque=1.0, J=0.05, h=0.02, h_max=10.0)
    sat = Satellite(mass=3.0, COM=np.array([0.04, -0.02, 0.05]),
                    J_0=_Jbody(), actuators=[rw])
    A = np.vstack([a.axis for a in sat.rw_actuators])

    x = np.hstack(([0.02, -0.015, 0.01], [1.0, 0.0, 0.0, 0.0], [0.02]))
    H0 = _Heci(sat, x, A)

    dt, N = 0.1, 3000
    u = np.zeros(1)  # zero motor command -> internal torque only; total H constant
    maxdH = 0.0
    for _ in range(N):
        x = sat.noiseless_rk4(x, u, dt, os0, os0, mid_orbital_state=os0)
        maxdH = max(maxdH, np.linalg.norm(_Heci(sat, x, A) - H0) / np.linalg.norm(H0))
    assert maxdH < 1e-5, f"angular momentum drift {maxdH:.2e}"


# --------------------------------------------------------------------------
# Bug 1c: dynamics derivative must equal the analytic Euler equation about COM
# --------------------------------------------------------------------------
@pytest.mark.parametrize("com", COM_OFFSETS, ids=COM_IDS)
def test_dynamics_core_matches_analytic_euler_com_offset(os0, com):
    sat = Satellite(mass=2.0, COM=com, J_0=_Jbody())
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(size=3) * 0.05
        q = np.array([1.0, 0.0, 0.0, 0.0])
        x = np.hstack((w, q))
        xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os0)
        wdot_analytic = sat.invJ_COM @ (-np.cross(w, sat.J_COM @ w))
        assert np.allclose(xdot[:3], wdot_analytic, atol=1e-10), \
            f"wdot {xdot[:3]} vs analytic {wdot_analytic}"


# --------------------------------------------------------------------------
# Bug 1d: analytic Jacobian must match a finite difference with a COM offset
# --------------------------------------------------------------------------
def test_dynjac_matches_fd_with_com_offset(os0):
    # dynJacCore needs >=1 actuator (np.vstack of an empty list raises); a single
    # MTQ with B=0 in os0 contributes zero torque, so the run stays torque-free
    # and the only thing under test is the gyroscopic J_COM vs J_0 term.
    from ADCS.satellite_hardware.actuators import MTQ
    mtq = MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)
    sat = Satellite(mass=4.0, COM=np.array([0.05, 0.02, -0.03]),
                    J_0=_Jbody(), actuators=[mtq])
    x = np.hstack(([0.03, -0.02, 0.012], [1.0, 0.0, 0.0, 0.0]))
    u = np.zeros(1)

    def f(xx):
        return sat.dynamics_core(x=xx, u=u, orbital_state=os0)

    Jac = sat.dynJacCore(x, u, os0)[0]  # dxdot/dx, shape (n, n) with state on axis 1
    eps = 1e-7
    fd = np.zeros((x.size, x.size))
    for i in range(x.size):
        dx = np.zeros(x.size); dx[i] = eps
        fd[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    # dynJacCore stores d(xdot)/dx with the perturbed-state index on axis 0.
    assert np.allclose(Jac.T[: x.size, : x.size], fd, atol=1e-4), \
        f"max |Jac - FD| = {np.abs(Jac.T[:x.size,:x.size]-fd).max():.2e}"


# --------------------------------------------------------------------------
# Bug 2: noiseless_rk4 must not mutate the caller's input array
# --------------------------------------------------------------------------
def test_noiseless_rk4_does_not_mutate_input(os0):
    sat = Satellite(J_0=_Jbody())
    x = np.array([0.01, -0.008, 0.006, 2.0, 0.0, 0.0, 0.0])  # deliberately non-unit q
    x_before = x.copy()
    sat.noiseless_rk4(x, np.zeros(0), 0.5, os0, os0, mid_orbital_state=os0)
    assert np.array_equal(x, x_before), \
        f"input mutated: {x_before} -> {x}"
>>>>>>> 3dd3be9 (moved tests and updated estimatedsatellite)
