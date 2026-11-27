"""Basilisk runner (run inside the Basilisk virtual environment).

Basilisk's ``Spacecraft`` always integrates the hub's rotational and
translational states together, with its default fixed-step RK4 integrator, so
the "orbit" scenario runs the same coupled model. J2 is applied through
Basilisk's spherical-harmonics gravity with a degree-2 coefficient file written
from the shared scenario constants (mu, R, J2), so the physics matches exactly.
"""

from __future__ import annotations

import math
import tempfile
from importlib.metadata import version
from pathlib import Path

from common import S, emit, parse_args, timed_runs

from Basilisk.architecture import messaging  # noqa: F401  (loads message types)
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.utilities import RigidBodyKinematics as rbk


def gravity_file(directory: Path) -> str:
    """Normalized JPL-format coefficient file with only C20 = -J2/sqrt(5)."""
    path = directory / "j2_only.csv"
    c20 = -S.J2 / math.sqrt(5.0)
    rows = [
        f"{S.R_EARTH},{S.MU},0.0,2,2,1,0.0,0.0",
        "0,0,1.0,0.0", "1,0,0.0,0.0", "1,1,0.0,0.0",
        f"2,0,{c20!r},0.0", "2,1,0.0,0.0", "2,2,0.0,0.0",
    ]
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def make_run_factory(grav_path: str):
    def make_run(steps: int):
        sim = SimulationBaseClass.SimBaseClass()
        proc = sim.CreateNewProcess("proc")
        proc.addTask(sim.CreateNewTask("task", macros.sec2nano(S.DT)))

        sc = spacecraft.Spacecraft()
        sc.ModelTag = "sat"
        sc.hub.mHub = S.MASS
        sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        d = S.INERTIA_DIAG
        sc.hub.IHubPntBc_B = [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
        sc.hub.r_CN_NInit = [[c] for c in S.R0]
        sc.hub.v_CN_NInit = [[c] for c in S.V0]
        sc.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # identity attitude
        sc.hub.omega_BN_BInit = [[c] for c in S.W0]

        factory = simIncludeGravBody.gravBodyFactory()
        earth = factory.createEarth()
        earth.isCentralBody = True
        earth.mu = S.MU
        earth.radEquator = S.R_EARTH
        earth.useSphericalHarmonicsGravityModel(grav_path, 2)
        factory.addBodiesTo(sc)

        sim.AddModelToTask("task", sc)
        rec = sc.scStateOutMsg.recorder()  # record every step
        sim.AddModelToTask("task", rec)
        sim.InitializeSimulation()
        sim.ConfigureStopTime(macros.sec2nano(steps * S.DT))

        def run():
            sim.ExecuteSimulation()
            sigma = list(rec.sigma_BN[-1])
            return {
                "r": list(rec.r_BN_N[-1]),
                "v": list(rec.v_BN_N[-1]),
                "dcm_bi": [list(row) for row in rbk.MRP2C(sigma)],
                "w": list(rec.omega_BN_B[-1]),
                "recorded": len(rec.times()),
            }
        return run
    return make_run


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        samples, final = timed_runs(make_run_factory(gravity_file(Path(tmp))), args)
    emit(tool="Basilisk", variant="Basilisk", version=version("bsk"),
         scenario=args.scenario, steps=args.steps, samples_s=samples, final=final,
         notes=["Spacecraft integrates attitude and orbit together (default RK4).",
                "J2 via degree-2 spherical harmonics written from the shared constants."])


if __name__ == "__main__":
    main()
