r"""
Low-thrust orbit adjustment: a commanded thrust acceleration (ECI / RTN / LVLH /
BODY) superposed onto the orbit via the external-acceleration channel, supplied
by a pluggable thrust source (open-loop schedule or closed-loop controller).
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import (
    SatelliteAgent, Constellation, FormationWorld,
    ConstantThrust, ScheduledThrust, CallableThrust, thrust_command_to_eci,
)
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.orbits.universal_constants import EarthConstants as E
from ADCS.simulate import simulate

EPHEM = Ephemeris()


# --------------------------------------------------------------------------- #
# Frame conversions
# --------------------------------------------------------------------------- #
def test_thrust_frame_conversions():
    R = np.array([7000.0, 0.0, 0.0])
    V = np.array([0.0, 7.546, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity attitude

    assert np.allclose(thrust_command_to_eci([1, 2, 3], "ECI", q, R, V), [1, 2, 3])
    # RTN: radial -> +x (zenith), along-track -> +y (velocity), cross -> +z (normal)
    assert np.allclose(thrust_command_to_eci([1, 0, 0], "RTN", q, R, V), [1, 0, 0])
    assert np.allclose(thrust_command_to_eci([0, 1, 0], "RTN", q, R, V), [0, 1, 0])
    assert np.allclose(thrust_command_to_eci([0, 0, 1], "RTN", q, R, V), [0, 0, 1])
    # BODY with a 90 deg rotation about z: body +x -> ECI +y
    qz = normalize(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
    got = thrust_command_to_eci([1, 0, 0], "BODY", qz, R, V)
    assert np.allclose(got, rot_mat(qz) @ np.array([1.0, 0, 0]))
    assert np.allclose(got, [0, 1, 0], atol=1e-9)

    with pytest.raises(ValueError):
        thrust_command_to_eci([1, 0, 0], "GALACTIC", q, R, V)


# --------------------------------------------------------------------------- #
# Physics: along-track thrust raises the semi-major axis
# --------------------------------------------------------------------------- #
def _circular_os(a0):
    v0 = np.sqrt(E.mu_e / a0)
    return Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([a0, 0.0, 0.0]), V=np.array([0.0, v0, 0.0]))


def _sma(R, V):
    return 1.0 / (2.0 / np.linalg.norm(R) - (V @ V) / E.mu_e)


def test_constant_along_track_thrust_raises_semimajor_axis():
    a0 = 7000.0
    n = np.sqrt(E.mu_e / a0**3)
    a_t = 1e-3  # m/s^2 along-track
    dt, tf = 5.0, 600.0
    agent = SatelliteAgent(
        x=np.concatenate([np.zeros(3), [1.0, 0, 0, 0]]), satellite=Satellite(mass=4.0),
        goal_list=GoalList({0.0: No_Goal()}), sat_id=0,
        thrust_source=ConstantThrust([0.0, a_t, 0.0], "RTN"),
    )
    out = Constellation([agent], [_circular_os(a0)], dt=dt, tf=tf, verbose=False).run()[0]
    af = _sma(out.os_hist[-1].R, out.os_hist[-1].V)
    da_theory = 2.0 * (a_t / 1000.0) * tf / n  # a_t [km/s^2] * 2/n * dt
    assert np.isclose(af - a0, da_theory, rtol=0.02)  # leading-order analytic match


def test_radial_thrust_does_not_secularly_change_energy():
    # Pure radial thrust does work that averages out over an orbit -> negligible
    # net semi-major-axis change relative to an along-track command of equal size.
    a0 = 7000.0
    os0 = _circular_os(a0)
    period = 2.0 * np.pi * np.sqrt(a0**3 / E.mu_e)
    dt = period / 400
    a_t = 1e-3

    def run(direction):
        ag = SatelliteAgent(x=np.concatenate([np.zeros(3), [1.0, 0, 0, 0]]), satellite=Satellite(mass=4.0),
                            goal_list=GoalList({0.0: No_Goal()}), sat_id=0,
                            thrust_source=ConstantThrust(direction, "RTN"))
        out = Constellation([ag], [_circular_os(a0)], dt=dt, tf=period, verbose=False).run()[0]
        return _sma(out.os_hist[-1].R, out.os_hist[-1].V) - a0

    da_radial = abs(run([a_t, 0.0, 0.0]))
    da_along = abs(run([0.0, a_t, 0.0]))
    assert da_radial < 0.1 * da_along


# --------------------------------------------------------------------------- #
# Superposition with aero
# --------------------------------------------------------------------------- #
def test_thrust_superposes_with_aero():
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    os = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([6678.0, 0.0, 0.0]), V=np.array([0.0, 7.726, 0.0]))
    agent = SatelliteAgent(
        x=np.concatenate([np.zeros(3), normalize(np.array([1.0, 0.2, -0.1, 0.05]))]), satellite=Satellite(mass=4.0),
        goal_list=GoalList({0.0: No_Goal()}), sat_id=0,
        aero_model=AeroModel(faces, areas, Cn=2.4, Ct=0.3),
        thrust_source=ConstantThrust([0.0, 1e-4, 0.0], "RTN"),
    )
    a_aero = agent.aero_accel_eci(os)
    a_thr = agent.thrust_accel_eci(os, 0.0, None)
    a_tot = agent.external_accel_eci(os, 0.0, None)
    assert np.allclose(a_tot, a_aero + a_thr)
    # aero excluded when the master switch is off, thrust still applied
    assert np.allclose(agent.external_accel_eci(os, 0.0, None, aero=False), a_thr)


# --------------------------------------------------------------------------- #
# Pluggable: open-loop schedule and closed-loop world-reading control
# --------------------------------------------------------------------------- #
def test_scheduled_thrust_selects_segment():
    sched = ScheduledThrust(times_J2000=[0.0, 1.0, 2.0],
                            accels=[[0, 1e-4, 0], [0, -1e-4, 0], [0, 0, 0]], frame="RTN")
    assert sched(-0.5, None, None, None) is None
    assert np.allclose(sched(0.5, None, None, None)[0], [0, 1e-4, 0])
    assert np.allclose(sched(1.5, None, None, None)[0], [0, -1e-4, 0])
    assert np.allclose(sched(9.0, None, None, None)[0], [0, 0, 0])


def _trailing(a0, gap_km):
    """A clean phase-shifted state trailing the reference by gap_km on the same orbit."""
    v0 = np.sqrt(E.mu_e / a0)
    dth = gap_km / a0
    c, s = np.cos(-dth), np.sin(-dth)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Orbital_State(ephem=EPHEM, J2000=0.22, R=Rz @ np.array([a0, 0.0, 0.0]), V=Rz @ np.array([0.0, v0, 0.0]))


def _along_track_unit(R, V):
    return np.cross(normalize(np.cross(R, V)), normalize(R))


def test_closed_loop_thrust_reads_world_and_reduces_drift():
    # Follower + leader on the same orbit (clean phase-shifted ICs); differential
    # drag drives them apart along-track. A correct-sign along-track controller
    # (retrograde to catch a receding leader) reads the leader from the world and
    # must substantially REDUCE the differential-drag drift vs free drift.
    a0 = 6678.0
    gap0 = 0.3  # km
    dt, tf = 15.0, 4.0 * 2.0 * np.pi * np.sqrt(a0**3 / E.mu_e)

    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    aero = lambda scale: AeroModel(faces, scale * areas, Cn=2.6, Ct=0.3)

    def controller(t, x, os, w):
        R = np.asarray(os.R); V = np.asarray(os.V); e_T = _along_track_unit(R, V)
        gap = float((w.position("lead") - R) @ e_T)
        gap_rate = float((w.velocity("lead") - V) @ e_T)
        return np.array([0.0, -(3e-6 * (gap - gap0) + 1.6e-2 * gap_rate), 0.0])  # m/s^2, retrograde sign

    def build(thrust):
        x0 = np.concatenate([np.zeros(3), [1.0, 0, 0, 0]])
        lead = SatelliteAgent(x=x0, satellite=Satellite(mass=4.0), goal_list=GoalList({0.22: No_Goal()}),
                              sat_id="lead", aero_model=aero(1.04))  # leader drags more -> drifts ahead
        foll = SatelliteAgent(x=x0, satellite=Satellite(mass=4.0), goal_list=GoalList({0.22: No_Goal()}),
                              sat_id="foll", aero_model=aero(1.0),
                              thrust_source=CallableThrust(controller, "RTN") if thrust else None)
        return [lead, foll]

    os_list = [_trailing(a0, 0.0), _trailing(a0, gap0)]

    def final_gap(agents):
        out = Constellation(agents, os_list, dt=dt, tf=tf, aero=True, world=FormationWorld(), verbose=False).run()
        Rl, Rf, Vf = out[0].os_hist[-1].R, out[1].os_hist[-1].R, out[1].os_hist[-1].V
        return abs(float((Rl - Rf) @ _along_track_unit(Rf, Vf)) - gap0) * 1e3  # m from target

    err_free = final_gap(build(False))
    err_ctrl = final_gap(build(True))
    # The closed-loop along-track law (reading the leader from the world)
    # substantially reduces the differential-drag drift. A simple P+D reduces it
    # ~35%; the demo's PID with an integral term does far better.
    assert err_ctrl < 0.8 * err_free


# --------------------------------------------------------------------------- #
# simulate() single-satellite thrust
# --------------------------------------------------------------------------- #
def test_simulate_single_sat_thrust_changes_orbit_and_matches_constellation():
    dt, tf = 10.0, 800.0
    a0 = 7000.0
    src = lambda: ConstantThrust([0.0, 5e-4, 0.0], "RTN")
    x0 = np.concatenate([np.zeros(3), [1.0, 0, 0, 0]])

    off = simulate(x=x0, satellite=Satellite(mass=4.0), goal=No_Goal(), os0=_circular_os(a0), dt=dt, tf=tf)[0]
    on = simulate(x=x0, satellite=Satellite(mass=4.0), goal=No_Goal(), os0=_circular_os(a0), dt=dt, tf=tf,
                  thrust_source=src())[0]
    assert np.linalg.norm(on.os_hist[-1].R - off.os_hist[-1].R) > 1e-2

    agent = SatelliteAgent(x=x0, satellite=Satellite(mass=4.0), goal_list=GoalList({0.0: No_Goal()}),
                           sat_id=0, thrust_source=src())
    con = Constellation([agent], [_circular_os(a0)], dt=dt, tf=tf, verbose=False).run()[0]
    sim_R = np.vstack([o.R for o in on.os_hist])
    con_R = np.vstack([o.R for o in con.os_hist])
    assert np.allclose(sim_R, con_R, rtol=1e-7, atol=1e-6)


def test_coast_returns_gravity_only():
    # A source that returns None (coast) leaves the orbit gravity-only.
    coast = lambda t, x, os, w: None
    a0 = 7000.0
    grav = simulate(x=np.concatenate([np.zeros(3), [1.0, 0, 0, 0]]), satellite=Satellite(mass=4.0),
                    goal=No_Goal(), os0=_circular_os(a0), dt=10.0, tf=300.0)[0]
    coasted = simulate(x=np.concatenate([np.zeros(3), [1.0, 0, 0, 0]]), satellite=Satellite(mass=4.0),
                       goal=No_Goal(), os0=_circular_os(a0), dt=10.0, tf=300.0,
                       thrust_source=CallableThrust(coast, "RTN"))[0]
    # in-loop path with zero thrust still reproduces the gravity orbit closely
    g_R = np.vstack([o.R for o in grav.os_hist])
    c_R = np.vstack([o.R for o in coasted.os_hist])
    assert np.allclose(g_R, c_R, rtol=1e-6, atol=1e-3)
