"""Orekit runner (run inside the Orekit virtual environment).

Orbit-only: Orekit models attitude kinematically (attitude laws) and does not
integrate rigid-body rotational dynamics, so it only runs the "orbit" scenario.
Uses a NumericalPropagator with Orekit's ClassicalRungeKuttaIntegrator (fixed
step RK4) on Cartesian coordinates, central gravity from the orbit's mu, and
``J2OnlyPerturbation`` built from the shared constants in an inertial frame
(J2 axis = inertial z). States are recorded every step by Orekit's own
ephemeris generator (on the Java side).
"""

from __future__ import annotations

import os
from importlib.metadata import version

import jdk4py

os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))

import orekit_jpype  # noqa: E402

orekit_jpype.initVM()

from common import S, emit, parse_args, timed_runs  # noqa: E402

from org.hipparchus.geometry.euclidean.threed import Vector3D  # noqa: E402
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator  # noqa: E402
from org.orekit.forces.gravity import J2OnlyPerturbation  # noqa: E402
from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import CartesianOrbit, OrbitType  # noqa: E402
from org.orekit.propagation import SpacecraftState  # noqa: E402
from org.orekit.propagation.numerical import NumericalPropagator  # noqa: E402
from org.orekit.time import AbsoluteDate  # noqa: E402
from org.orekit.utils import PVCoordinates  # noqa: E402

JVM_WARMUP_STEPS = 10_000


def make_run(steps: int):
    frame = FramesFactory.getEME2000()
    epoch = AbsoluteDate.J2000_EPOCH
    orbit = CartesianOrbit(PVCoordinates(Vector3D(*S.R0), Vector3D(*S.V0)), frame, epoch, S.MU)
    propagator = NumericalPropagator(ClassicalRungeKuttaIntegrator(S.DT))
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setMu(S.MU)
    propagator.addForceModel(J2OnlyPerturbation(S.MU, S.R_EARTH, S.J2, frame))
    propagator.setInitialState(SpacecraftState(orbit).withMass(S.MASS))
    generator = propagator.getEphemerisGenerator()  # record the trajectory
    target = epoch.shiftedBy(steps * S.DT)

    def run():
        state = propagator.propagate(target)
        generator.getGeneratedEphemeris()
        pv = state.getPVCoordinates(frame)
        p, v = pv.getPosition(), pv.getVelocity()
        return {"r": [p.getX(), p.getY(), p.getZ()], "v": [v.getX(), v.getY(), v.getZ()]}
    return run


def main():
    args = parse_args()
    if args.scenario != "orbit":
        emit(tool="Orekit", variant="Orekit", version=version("orekit-jpype"),
             scenario=args.scenario, steps=args.steps, skipped=True, samples_s=[], final=None,
             notes=["Orekit does not integrate rigid-body attitude dynamics."])
        return
    args.warmup_steps = max(args.warmup_steps, JVM_WARMUP_STEPS)
    samples, final = timed_runs(make_run, args)
    emit(tool="Orekit", variant="Orekit", version=version("orekit-jpype"),
         scenario=args.scenario, steps=args.steps, samples_s=samples, final=final,
         notes=["Orbit only; fixed-step classical RK4 on Cartesian coordinates.",
                f"JVM warm-up of {JVM_WARMUP_STEPS} steps before timing."])


if __name__ == "__main__":
    main()
