r"""
Low-thrust along-track station-keeping demo.

A leader + follower in a ~300 km LEO formation. The two satellites have slightly
different ballistic configurations (2% drag-area difference), so *differential
drag* slowly drives them apart along-track. The follower runs a simple closed-
loop along-track thrust law that reads the leader's state from the shared
FormationWorld and substantially reduces the drift.

This exercises the general low-thrust orbit-control hook:
  * thrust commanded in the RTN/LVLH frame (along-track),
  * a pluggable closed-loop controller reading neighbour state from FormationWorld,
  * superposed with attitude-coupled aerodynamic drag+lift on the orbit.

Note the (correct, counter-intuitive) orbital mechanics: to *catch up* in-track
the follower thrusts RETROGRADE (lowering its orbit shortens the period). The PID
here is a deliberately simple starting point -- tight station-keeping is itself a
control-design problem (the kind this simulator is meant to help study).

Run:  python examples/formation/low_thrust_stationkeeping_demo.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np

from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation, FormationWorld, CallableThrust
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.orbits.universal_constants import EarthConstants as E

EPHEM = Ephemeris()
ALT = 6678.0                       # ~300 km radius
TARGET_GAP_KM = 0.30               # desired along-track gap behind the leader
V0 = np.sqrt(E.mu_e / ALT)
PERIOD = 2.0 * np.pi * np.sqrt(ALT**3 / E.mu_e)


def along_track_unit(R, V):
    return np.cross(normalize(np.cross(R, V)), normalize(R))


def trailing_state(gap_km):
    """A point trailing the leader by ``gap_km`` along-track on the SAME orbit."""
    dth = gap_km / ALT
    c, s = np.cos(-dth), np.sin(-dth)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Orbital_State(ephem=EPHEM, J2000=0.22,
                         R=Rz @ np.array([ALT, 0.0, 0.0]),
                         V=Rz @ np.array([0.0, V0, 0.0]))


def aero(area_scale):
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = area_scale * np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    return AeroModel(faces, areas, Cn=2.6, Ct=0.3)


def along_track_pid(Kp=3e-6, Ki=3e-10, Kd=1.6e-2):
    """Closed-loop along-track station-keeping (PID on the along-track gap)."""
    state = {"integral": 0.0, "t_prev": None}

    def law(t_J2000, x, os, world):
        R = np.asarray(os.R); V = np.asarray(os.V)
        e_T = along_track_unit(R, V)
        gap = float((world.position("lead") - R) @ e_T)            # km, +ve leader ahead
        gap_rate = float((world.velocity("lead") - V) @ e_T)        # km/s
        dt = 0.0 if state["t_prev"] is None else (t_J2000 - state["t_prev"]) * 3.15576e9
        state["t_prev"] = t_J2000
        state["integral"] += (gap - TARGET_GAP_KM) * dt
        # Retrograde (negative along-track) to catch up to a receding leader.
        accel = -(Kp * (gap - TARGET_GAP_KM) + Kd * gap_rate + Ki * state["integral"])
        return np.array([0.0, float(np.clip(accel, -5e-4, 5e-4)), 0.0])

    return law


def build(thrust_on):
    world = FormationWorld()
    x0 = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])
    lead = SatelliteAgent(x=x0, satellite=Satellite(mass=4.0), goal_list=GoalList({0.22: No_Goal()}),
                          sat_id="lead", aero_model=aero(1.02))   # 2% more drag area
    foll = SatelliteAgent(x=x0, satellite=Satellite(mass=4.0), goal_list=GoalList({0.22: No_Goal()}),
                          sat_id="foll", aero_model=aero(1.00),
                          thrust_source=CallableThrust(along_track_pid(), "RTN") if thrust_on else None)
    os_list = [trailing_state(0.0), trailing_state(TARGET_GAP_KM)]
    return [lead, foll], os_list, world


def gap_history(out):
    Rl = np.vstack([o.R for o in out[0].os_hist])
    Rf = np.vstack([o.R for o in out[1].os_hist])
    Vf = np.vstack([o.V for o in out[1].os_hist])
    return np.array([float((Rl[k] - Rf[k]) @ along_track_unit(Rf[k], Vf[k]))
                     for k in range(len(Rl))]) * 1e3  # m


def main():
    dt, tf = 15.0, 6.0 * PERIOD

    agents, os_list, _ = build(thrust_on=False)
    free = Constellation(agents, os_list, dt=dt, tf=tf, aero=True, verbose=True).run()

    agents, os_list, world = build(thrust_on=True)
    ctrl = Constellation(agents, os_list, dt=dt, tf=tf, aero=True, world=world, verbose=True).run()

    g_free = gap_history(free)
    g_ctrl = gap_history(ctrl)
    print(f"\nAlong-track gap behind leader over {tf/PERIOD:.0f} orbits "
          f"(target {TARGET_GAP_KM*1e3:.0f} m, 2% differential drag):")
    print(f"  free drift : {g_free[0]:5.0f} m -> {g_free[-1]:7.0f} m")
    print(f"  controlled : {g_ctrl[0]:5.0f} m -> {g_ctrl[-1]:7.0f} m  "
          f"(range {g_ctrl.min():.0f}-{g_ctrl.max():.0f} m)")
    print(f"  -> low-thrust keeps the formation ~{(g_free[-1]-300)/max(1.0,(g_ctrl.max()-300)):.0f}x tighter")

    try:
        import matplotlib.pyplot as plt
        t = np.linspace(0, tf / PERIOD, len(g_free))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, g_free, label="free drift (differential drag)")
        ax.plot(t, g_ctrl, label="low-thrust station-keeping")
        ax.axhline(TARGET_GAP_KM * 1e3, color="k", ls="--", lw=1, label="target gap")
        ax.set_xlabel("orbits"); ax.set_ylabel("along-track gap [m]")
        ax.set_title("Low-thrust along-track station-keeping vs free differential-drag drift")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stationkeeping.png"), dpi=110)
        print("saved stationkeeping.png")
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
