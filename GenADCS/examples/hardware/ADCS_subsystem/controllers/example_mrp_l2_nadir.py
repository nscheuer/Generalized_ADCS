"""3-axis Nadir tracking with AttitudeController (Passthrough + MRPFeedbackLinearized + L2Allocator).

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/ADCS_subsystem/controllers/example_mrp_l2_nadir.py``
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    AttitudeController,
    L2Allocator,
    MRPFeedbackLinearized,
    NoDesaturation,
    Passthrough,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import NadirGoal
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
    BodyAngularRatePlot,
    CommandedWheelTorquePlot,
    GoalAttitudePlot,
    GoalRatePlot,
    ReferenceTorquePlot,
    plot,
)


def main(show: bool = True):
    dt = 0.05
    tf = 40.0

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.15, 0.12, 0.08)),
        panels=[],
    )
    wheels = [
        ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.2, 1.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e-3, 0.2, 1.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-3, 0.2, 1.0),
    ]
    controller = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.5, kd=1.4),
        feedforward=Passthrough(),
        allocator=L2Allocator(),
        desaturation=NoDesaturation(),
    )
    adcs = ADCSSubsystem(
        reaction_wheels=wheels,
        N=1,
        state_machine=ADCSStateMachine(
            initial="nadir",
            modes=[ADCSMode("nadir", goal=NadirGoal(), controller=controller)],
        ),
    )
    satellite = Satellite(geometry=geometry, adcs=adcs)
    environment = Environment(planetary_body=Earth())
    dynamics = DynamicsConfig(forces=[], torques=[])

    q0 = np.array([np.cos(np.pi / 6.0), np.sin(np.pi / 6.0), 0.0, 0.0])
    state = State(
        attitude=AttitudeState(orientation=q0, rates=(0.04, -0.03, 0.02)),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
        ),
        internal=WheelMomentumState(wheel_momentum=np.zeros(3)),
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
            satellite_rate=True,
            goal_attitude=True,
            goal_rate=True,
            reference_torque=True,
            commanded_wheel_torque=True,
        ),
    )
    sat_res = result.satellites["sat_000"]
    print(
        f"Nadir tracking final |q · q_goal| = "
        f"{abs(np.dot(sat_res.satellite_attitude[-1], sat_res.goal_attitude[-1])):.6f}"
    )

    plot(
        result,
        AttitudePlot(),
        GoalAttitudePlot(),
        BodyAngularRatePlot(),
        GoalRatePlot(),
        ReferenceTorquePlot(),
        CommandedWheelTorquePlot(),
        AbsolutePointingErrorPlot(option="magnitude"),
        layout=(4, 2),
        title="MRP Feedback Linearized + L2 Allocator (Nadir Goal)",
        show=show,
    )
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result


if __name__ == "__main__":
    main()
