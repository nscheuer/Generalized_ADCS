"""End-to-end tests for AttitudeController, ResultsFormat, and plotting."""

import numpy as np

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    AttitudeController,
    BDot,
    HoganSchaubManagement,
    L2Allocator,
    LInfinityAllocator,
    Magnetorquer,
    MRPFeedbackLinearized,
    NoDesaturation,
    Passthrough,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import NadirGoal, QuaternionGoal
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    Gyroscope,
    Magnetometer,
)
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


def test_attitude_controller_tracks_quaternion_goal() -> None:
    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.15, 0.12, 0.08)), panels=[])
    wheels = [
        ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.2, 1.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e-3, 0.2, 1.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-3, 0.2, 1.0),
    ]
    ctrl = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.5, kd=1.4),
        feedforward=Passthrough(),
        allocator=L2Allocator(),
        desaturation=NoDesaturation(),
    )
    target_q = np.array([1.0, 0.0, 0.0, 0.0])
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            reaction_wheels=wheels,
            N=1,
            state_machine=ADCSStateMachine(
                initial="track",
                modes=[ADCSMode("track", goal=QuaternionGoal(target_q), controller=ctrl)],
            ),
        ),
    )
    env = Environment(planetary_body=Earth())
    q0 = np.array([np.cos(np.pi / 6.0), np.sin(np.pi / 6.0), 0.0, 0.0])
    state = State(
        attitude=AttitudeState(q0, (0.05, -0.04, 0.03)),
        orbit=OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5
        ),
        internal=WheelMomentumState(wheel_momentum=np.zeros(3)),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.05)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.05)),
        tf=20.0,
    )
    res = simulate(
        sat,
        env,
        DynamicsConfig([], []),
        state,
        solver,
        resultsformat=ResultsFormat(
            N=2,
            satellite_attitude=True,
            satellite_rate=True,
            reference_torque=True,
            commanded_wheel_torque=True,
        ),
    )
    sat_res = res.satellites["sat_000"]
    assert abs(np.dot(sat_res.satellite_attitude[-1], target_q)) > 0.999
    assert np.linalg.norm(sat_res.satellite_rate[-1]) < 1e-3


def test_attitude_controller_4_wheels_linfinity_and_hogan_schaub() -> None:
    s3 = 1.0 / np.sqrt(3.0)
    a_mat = np.array(
        [[1.0, 0.0, 0.0, s3], [0.0, 1.0, 0.0, s3], [0.0, 0.0, 1.0, s3]],
        dtype=np.float64,
    )
    _, _, vh = np.linalg.svd(a_mat, full_matrices=True)
    null_vec = vh[3:].T[:, 0]

    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.15, 0.12, 0.08)), panels=[])
    wheels = [
        ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-3, 0.2, 2.0),
        ReactionWheel((0.0, 0.0, 0.0), (s3, s3, s3), 1e-3, 0.2, 2.0),
    ]
    ctrl = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.5, kd=1.4),
        allocator=LInfinityAllocator(),
        desaturation=HoganSchaubManagement(gain=0.5, target_momentum=0.0),
    )
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            reaction_wheels=wheels,
            N=1,
            state_machine=ADCSStateMachine(
                initial="track",
                modes=[ADCSMode("track", goal=QuaternionGoal((1.0, 0.0, 0.0, 0.0)), controller=ctrl)],
            ),
        ),
    )
    env = Environment(planetary_body=Earth())
    h0 = 0.4 * null_vec
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        orbit=OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5
        ),
        internal=WheelMomentumState(wheel_momentum=h0),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.05)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.05)),
        tf=15.0,
    )
    res = simulate(
        sat,
        env,
        DynamicsConfig([], []),
        state,
        solver,
        resultsformat=ResultsFormat(N=2, satellite_wheel_momenta=True),
    )
    h_final = res.satellites["sat_000"].satellite_wheel_momenta[-1]
    assert np.linalg.norm(h_final) < 0.05 * np.linalg.norm(h0)


def test_control_and_goal_resultsformat_and_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from GenADCS.backends.shared.plotting import (
        CommandedMagneticDipolePlot,
        CommandedWheelTorquePlot,
        FeedforwardAttitudePlot,
        FeedforwardRatePlot,
        FeedforwardTorquePlot,
        GoalAttitudePlot,
        GoalRatePlot,
        ReferenceTorquePlot,
        plot,
    )

    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.1, 0.1, 0.1)), panels=[])
    wheels = [ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.1, 1.0)]
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
        noise=1e-18 * np.eye(3),
        bias=np.zeros(3),
    )
    gyro = Gyroscope(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=1e-18 * np.eye(3),
        bias=np.zeros(3),
    )
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            reaction_wheels=wheels,
            magnetorquers=mtqs,
            sensors=[mag, gyro],
            N=1,
            state_machine=ADCSStateMachine(
                initial="nadir",
                modes=[ADCSMode("nadir", goal=NadirGoal(), controller=BDot(1e4))],
            ),
        ),
    )
    env = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.02, -0.01, 0.01)),
        orbit=OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 5_382.9, 5_382.9), 2_461_000.5
        ),
        internal=WheelMomentumState(wheel_momentum=np.zeros(1)),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.5)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.5)),
        tf=5.0,
    )
    rf = ResultsFormat(
        N=2,
        goal_attitude=True,
        goal_rate=True,
        feedforward_torque=True,
        feedforward_attitude=True,
        feedforward_rate=True,
        reference_torque=True,
        commanded_wheel_torque=True,
        commanded_magnetic_dipole=True,
    )
    res = simulate(sat, env, DynamicsConfig([], []), state, solver, resultsformat=rf)
    sat_res = res.satellites["sat_000"]
    assert sat_res.goal_attitude.shape == (len(sat_res.times_s), 4)
    assert np.all(np.isfinite(sat_res.goal_attitude))
    assert sat_res.goal_rate.shape == (len(sat_res.times_s), 3)
    assert sat_res.commanded_wheel_torque.shape == (len(sat_res.times_s), 1)
    assert sat_res.commanded_magnetic_dipole.shape == (len(sat_res.times_s), 3)
    assert np.any(sat_res.commanded_magnetic_dipole != 0.0)

    fig, _ = plot(
        res,
        GoalAttitudePlot(),
        GoalRatePlot(),
        FeedforwardTorquePlot(),
        FeedforwardAttitudePlot(),
        FeedforwardRatePlot(),
        ReferenceTorquePlot(),
        CommandedWheelTorquePlot(),
        CommandedMagneticDipolePlot(),
        layout=(4, 2),
    )
    plt.close(fig)
