r"""
Formation-flying demo: a small constellation simulated in one process.

Demonstrates the multi-satellite (formation) capability:
  * many satellites stepped on a shared clock with a batched environment,
  * formation-aware goals (each satellite points a boresight at its neighbour),
  * attitude-coupled orbital aerodynamics (drag + lift), so a satellite's
    attitude changes its own orbit -- the basis of aerodynamic formation control,
  * per-satellite seeded RNG (independent, reproducible runs).

Run:  python examples/formation/formation_flying_demo.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np

from ADCS.CONOPS.goals import Relative_Pointing_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation, FormationWorld
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.satellite_hardware.satellite import Satellite


def box_aero_model():
    """A simple asymmetric box so attitude visibly affects the aero force."""
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    return AeroModel(faces, areas, Cn=2.4, Ct=0.4)


def build_formation(n=5, altitude_km=400.0, spread_km=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ephem = Ephemeris()
    world = FormationWorld()

    r_circ = 6378.137 + altitude_km
    v_circ = np.sqrt(398600.4415 / r_circ)
    base_R = np.array([r_circ, 0.0, 0.0])
    base_V = np.array([0.0, v_circ, 0.0])

    agents, os0_list = [], []
    for i in range(n):
        offset = rng.normal(scale=spread_km, size=3)
        offset[0] = 0.0  # spread in-plane / cross-track, not radially
        os0_list.append(Orbital_State(ephem=ephem, J2000=0.22,
                                      R=base_R + offset, V=base_V, rho=5e-11))

        sat = Satellite(mass=4.0, J_0=np.diagflat([0.02, 0.03, 0.04]),
                        disturbances=[GG_Disturbance()]).seed(seed * 1000 + i)
        x0 = np.concatenate([rng.normal(scale=0.005, size=3), normalize(rng.normal(size=4))])

        # Each satellite points at the next one in a ring.
        goal = Relative_Pointing_Goal(world, target_id=(i + 1) % n)
        agents.append(SatelliteAgent(x=x0, satellite=sat,
                                     goal_list=GoalList({0.22: goal}),
                                     sat_id=i, aero_model=box_aero_model()))

    return agents, os0_list, world, ephem


def main():
    n = 5
    dt, tf = 5.0, 600.0
    agents, os0_list, world, ephem = build_formation(n=n)

    con = Constellation(agents, os0_list, dt=dt, tf=tf, world=world,
                        aero=True, zonal_order=4, verbose=True)
    results = con.run()

    print(f"\nFormation of {n} satellites, dt={dt}s, tf={tf}s "
          f"(aero drag+lift on, J2-J4 gravity)\n")

    # Relative separations between consecutive ring members over time.
    R = [np.vstack([os.R for os in run.os_hist]) for run in results.runs]
    for i in range(n):
        j = (i + 1) % n
        sep0 = np.linalg.norm(R[i][0] - R[j][0])
        sepf = np.linalg.norm(R[i][-1] - R[j][-1])
        print(f"  sat {i} <-> sat {j}: separation {sep0*1e3:7.1f} m -> {sepf*1e3:7.1f} m")

    # Final attitude rates (formation pointing should keep them bounded).
    rates = [float(np.linalg.norm(np.asarray(run.state_hist)[-1, 0:3])) for run in results.runs]
    print(f"\n  final body rates [rad/s]: " + ", ".join(f"{r:.2e}" for r in rates))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for i in range(n):
            ax.plot(R[i][:, 1] - R[0][:, 1], R[i][:, 2] - R[0][:, 2], label=f"sat {i}")
        ax.set_xlabel("along-track relative to sat 0 [km]")
        ax.set_ylabel("cross-track relative to sat 0 [km]")
        ax.set_title("Relative motion (aero drag+lift on)")
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(plotting skipped: {exc})")


if __name__ == "__main__":
    main()
