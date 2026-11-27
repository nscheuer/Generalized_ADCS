"""Multi-mode ADCS timeline: BDot detumble -> Sun pointing -> Nadir tracking.

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/ADCS_subsystem/controllers/example_mode_switching_attitude.py``
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import DE421, Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    AttitudeController,
    BDot,
    L2Allocator,
    MRPFeedbackLinearized,
    Magnetorquer,
    Passthrough,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import (
    NadirGoal,
    NoGoal,
    SunGoal,
)
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
    BodyAngularRatePlot,
    CommandedMagneticDipolePlot,
    CommandedWheelTorquePlot,
    GoalAttitudePlot,
    ReferenceTorquePlot,
    plot,
)


def main(show: bool = True):
    dt = 0.1
    tf = 90.0

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
    mtqs = [
        Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5),
    ]
    mag = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-16 * np.eye(3),
        bias=np.zeros(3),
    )
    att_ctrl = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.5, kd=1.4),
        feedforward=Passthrough(),
        allocator=L2Allocator(),
    )
    adcs = ADCSSubsystem(
        reaction_wheels=wheels,
        magnetorquers=mtqs,
        sensors=[mag],
        N=1,
        state_machine=ADCSStateMachine(
            initial="detumble",
            modes=[
                ADCSMode("detumble", goal=NoGoal(), controller=BDot(gain=5e4)),
                ADCSMode("sun_point", goal=SunGoal(), controller=att_ctrl),
                ADCSMode("nadir_point", goal=NadirGoal(), controller=att_ctrl),
            ],
            timeline=[
                (20.0, "sun_point"),
                (55.0, "nadir_point"),
            ],
        ),
    )
    satellite = Satellite(geometry=geometry, adcs=adcs)
    environment = Environment(
        planetary_body=Earth(), ephemeris=DE421(), magnetic_field=IGRF14()
    )
    dynamics = DynamicsConfig(forces=[], torques=[])

    state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0),
            rates=(0.06, -0.04, 0.03),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 5_382.9, 5_382.9),
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
            N=5,
            satellite_attitude=True,
            satellite_rate=True,
            goal_attitude=True,
            reference_torque=True,
            commanded_wheel_torque=True,
            commanded_magnetic_dipole=True,
        ),
    )
    sat_res = result.satellites["sat_000"]
    print(
        f"Mode-switching end |q · q_nadir| = "
        f"{abs(np.dot(sat_res.satellite_attitude[-1], sat_res.goal_attitude[-1])):.6f}"
    )

    plot(
        result,
        AttitudePlot(),
        GoalAttitudePlot(),
        AbsolutePointingErrorPlot(option="magnitude"),
        BodyAngularRatePlot(),
        ReferenceTorquePlot(),
        CommandedWheelTorquePlot(),
        CommandedMagneticDipolePlot(),
        layout=(4, 2),
        title="ADCS Timeline: BDot -> Sun Pointing -> Nadir Tracking",
        show=show,
    )
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result


if __name__ == "__main__":
    main()
