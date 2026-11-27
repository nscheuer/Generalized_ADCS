"""4-wheel pyramid with LInfinityAllocator and HoganSchaubManagement desaturation.

Includes both nullspace wheel redistribution and magnetorquer momentum dumping.

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/ADCS_subsystem/controllers/example_4rw_linf_desat.py``
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    AttitudeController,
    HoganSchaubManagement,
    LInfinityAllocator,
    MRPFeedbackLinearized,
    Magnetorquer,
    Passthrough,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import QuaternionGoal
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Magnetometer
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import (
    AttitudeState,
    OrbitalState,
    State,
    WheelMomentumState,
)
from GenADCS.backends.shared.plotting import (
    AbsolutePointingErrorPlot,
    AttitudePlot,
    CommandedMagneticDipolePlot,
    CommandedWheelTorquePlot,
    ReferenceTorquePlot,
    WheelMomentumPlot,
    plot,
)


def main(show: bool = True):
    dt = 0.05
    tf = 40.0
    s3 = 1.0 / np.sqrt(3.0)

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.15, 0.12, 0.08)),
        panels=[],
    )
    wheels = [
        ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (s3, s3, s3), 1e-3, 0.2, 2.0),
    ]
    mtqs = [
        Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0),
    ]
    mag = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-16 * np.eye(3),
        bias=np.zeros(3),
    )
    controller = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.5, kd=1.4),
        feedforward=Passthrough(),
        allocator=LInfinityAllocator(),
        desaturation=HoganSchaubManagement(
            gain=0.5, target_momentum=0.0, mtq_gain=0.2
        ),
    )
    adcs = ADCSSubsystem(
        reaction_wheels=wheels,
        magnetorquers=mtqs,
        sensors=[mag],
        N=1,
        state_machine=ADCSStateMachine(
            initial="point",
            modes=[
                ADCSMode(
                    "point",
                    goal=QuaternionGoal((1.0, 0.0, 0.0, 0.0)),
                    controller=controller,
                )
            ],
        ),
    )
    satellite = Satellite(geometry=geometry, adcs=adcs)
    environment = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    dynamics = DynamicsConfig(forces=[], torques=[])

    # Start with both nullspace and net wheel momentum + a 45-deg attitude offset.
    a_mat = np.array(
        [[1.0, 0.0, 0.0, s3], [0.0, 1.0, 0.0, s3], [0.0, 0.0, 1.0, s3]],
        dtype=np.float64,
    )
    _, _, vh = np.linalg.svd(a_mat, full_matrices=True)
    h0 = 0.35 * vh[3:].T[:, 0] + np.array([0.04, -0.03, 0.02, 0.01])
    q0 = np.array([np.cos(np.pi / 8.0), np.sin(np.pi / 8.0), 0.0, 0.0])

    state = State(
        attitude=AttitudeState(orientation=q0, rates=(0.02, -0.02, 0.01)),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 5_382.9, 5_382.9),
            julian_date=2_461_000.5,
        ),
        internal=WheelMomentumState(wheel_momentum=h0),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=dt)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=dt)),
        tf=tf,
    )

    result = simulate(
        satellite,
        environment,
        dynamics,
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(
            N=2,
            satellite_attitude=True,
            goal_attitude=True,
            satellite_wheel_momenta=True,
            reference_torque=True,
            commanded_wheel_torque=True,
            commanded_magnetic_dipole=True,
        ),
    )
    h = result.satellites["sat_000"].satellite_wheel_momenta
    print(f"Wheel momentum ||h||: {np.linalg.norm(h[0]):.4f} -> {np.linalg.norm(h[-1]):.4f} N m s")

    plot(
        result,
        AttitudePlot(),
        AbsolutePointingErrorPlot(option="magnitude"),
        ReferenceTorquePlot(),
        CommandedWheelTorquePlot(),
        WheelMomentumPlot(),
        CommandedMagneticDipolePlot(),
        layout=(3, 2),
        title="4-Wheel L-Infinity Allocator + Hogan-Schaub Desaturation (Nullspace + MTQ)",
        show=show,
    )
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result


if __name__ == "__main__":
    main()
