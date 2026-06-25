r"""
Phase 4: attitude-coupled orbital aerodynamics (drag + lift).

Covers the free-molecular panel force model, the ``external_accel`` hook in the
orbit dynamics (gravity-only stays bit-identical), and the end-to-end coupling
in Constellation: a satellite's *attitude* changes its *orbit* only when aero is
enabled (the defining signature of aerodynamic lift/drag control).
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.orbits.universal_constants import EarthConstants as E

EPHEM = Ephemeris()


def _drag_lift(F, vhat):
    drag = -float(np.dot(F, vhat))
    lift = float(np.linalg.norm(F - np.dot(F, vhat) * vhat))
    return drag, lift


# --------------------------------------------------------------------------- #
# Panel force model
# --------------------------------------------------------------------------- #
def test_zero_lift_at_normal_incidence():
    m = AeroModel(normals=[[1, 0, 0]], areas=[1.0], Cn=2.0, Ct=0.0)
    F = m.force_body([7600.0, 0.0, 0.0], 1e-11)
    d, l = _drag_lift(F, np.array([1.0, 0.0, 0.0]))
    assert d > 0.0
    assert l < 1e-12 * d


def test_lift_present_at_oblique_incidence():
    m = AeroModel(normals=[[1, 0, 0]], areas=[1.0], Cn=2.0, Ct=0.0)
    vhat = normalize(np.array([1.0, 1.0, 0.0]))
    F = m.force_body(7600.0 * vhat, 1e-11)
    d, l = _drag_lift(F, vhat)
    assert d > 0.0 and l > 0.0


def test_cn_equals_ct_is_lift_free_at_all_incidence():
    m = AeroModel(normals=[[1, 0, 0]], areas=[1.0], Cn=2.0, Ct=2.0)
    for ang in (10.0, 35.0, 60.0):
        a = np.radians(ang)
        vhat = np.array([np.cos(a), np.sin(a), 0.0])
        F = m.force_body(7600.0 * vhat, 1e-11)
        d, l = _drag_lift(F, vhat)
        assert l < 1e-9 * d  # pure drag


def test_drag_always_opposes_velocity_for_a_box():
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.03, 0.03, 0.06, 0.06, 0.02, 0.02])
    m = AeroModel(faces, areas, Cn=2.2, Ct=0.5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        vhat = normalize(rng.normal(size=3))
        F = m.force_body(7600.0 * vhat, 1e-11)
        d, _ = _drag_lift(F, vhat)
        assert d >= -1e-15  # net force never adds energy along +v


def test_wake_faces_contribute_nothing():
    m = AeroModel(normals=[[1, 0, 0]], areas=[1.0], Cn=2.0, Ct=0.0)
    assert np.allclose(m.force_body([-7600.0, 0.0, 0.0], 1e-11), 0.0)
    assert np.allclose(m.force_body([7600.0, 0.0, 0.0], 0.0), 0.0)  # zero density


# --------------------------------------------------------------------------- #
# external_accel hook
# --------------------------------------------------------------------------- #
def test_external_accel_none_is_bit_identical():
    R = np.array([6878.0, 100.0, -200.0])
    V = np.array([0.5, 7.6, 0.3])
    base = Orbital_State._orbit_dynamics_raw(R, V, E.mu_e, E.R_e, E.J2coeff, True)[1]
    same = Orbital_State._orbit_dynamics_raw(R, V, E.mu_e, E.R_e, E.J2coeff, True, external_accel=None)[1]
    assert np.array_equal(base, same)


def test_external_accel_adds_exactly_to_vdot():
    R = np.array([6878.0, 0.0, 0.0])
    V = np.array([0.0, 7.6, 0.0])
    a = np.array([1e-9, -2e-9, 3e-9])
    base = Orbital_State._orbit_dynamics_raw(R, V, E.mu_e, E.R_e, E.J2coeff, True)[1]
    with_ext = Orbital_State._orbit_dynamics_raw(R, V, E.mu_e, E.R_e, E.J2coeff, True, external_accel=a)[1]
    # Exact algebraically; tolerance covers float cancellation against the ~8e-3
    # gravity term (ulp ~ 1e-18), far below the 1e-9 signal.
    assert np.allclose(with_ext - base, a, rtol=1e-6, atol=1e-15)


# --------------------------------------------------------------------------- #
# End-to-end coupling in Constellation
# --------------------------------------------------------------------------- #
def _box_sat():
    return Satellite(mass=4.0, J_0=np.diagflat([0.02, 0.03, 0.04]))


def _box_aero():
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])  # asymmetric -> orientation matters
    return AeroModel(faces, areas, Cn=2.6, Ct=0.3)


def _leo_os(R=None, V=None):
    # ~300 km circular-ish, where drag/lift are appreciable
    R = np.array([6678.0, 0.0, 0.0]) if R is None else np.asarray(R, dtype=float)
    V = np.array([0.0, 7.726, 0.0]) if V is None else np.asarray(V, dtype=float)
    return Orbital_State(ephem=EPHEM, J2000=0.22, R=R, V=V, rho=5e-11)


def _agent(q, aero=True, sat_id=0):
    x0 = np.concatenate([np.zeros(3), normalize(q)])  # zero rate -> attitude held fixed
    return SatelliteAgent(
        x=x0, satellite=_box_sat(),
        goal_list=GoalList({0.22: No_Goal()}), sat_id=sat_id,
        aero_model=_box_aero() if aero else None,
    )


def test_aero_perturbs_the_orbit():
    dt, tf = 10.0, 1500.0
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out_off = Constellation([_agent(q, aero=True)], [_leo_os()], dt=dt, tf=tf, aero=False, verbose=False).run()
    out_on = Constellation([_agent(q, aero=True)], [_leo_os()], dt=dt, tf=tf, aero=True, verbose=False).run()
    R_off = out_off[0].os_hist[-1].R
    R_on = out_on[0].os_hist[-1].R
    assert np.linalg.norm(R_on - R_off) > 1e-3  # aero measurably changes the trajectory


def test_attitude_changes_orbit_only_through_aero():
    dt, tf = 10.0, 2000.0
    # Two satellites, identical orbit initial conditions, DIFFERENT fixed attitudes.
    qA = np.array([1.0, 0.0, 0.0, 0.0])
    qB = normalize(np.array([np.cos(np.radians(35)), 0.0, 0.0, np.sin(np.radians(35))]))  # 70 deg about z

    def run(aero):
        agents = [_agent(qA, aero=aero, sat_id="A"), _agent(qB, aero=aero, sat_id="B")]
        out = Constellation(agents, [_leo_os(), _leo_os()], dt=dt, tf=tf, aero=aero, verbose=False).run()
        return out[0].os_hist[-1].R, out[1].os_hist[-1].R

    # Aero OFF: gravity is attitude-independent -> identical trajectories.
    RA_off, RB_off = run(False)
    assert np.allclose(RA_off, RB_off, atol=1e-9)

    # Aero ON: different attitudes -> different aero force -> different orbits.
    RA_on, RB_on = run(True)
    assert np.linalg.norm(RA_on - RB_on) > 1e-3
