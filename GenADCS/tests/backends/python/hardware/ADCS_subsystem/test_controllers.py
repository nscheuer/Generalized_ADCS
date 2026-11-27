"""Tests for Controller base class, BDot, and BCross."""

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSSubsystem,
    BCross,
    BDot,
    Controller,
    Magnetorquer,
    NoController,
    Satellite,
    SatelliteGeometry,
)
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


def _make_mag() -> Magnetometer:
    return Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-18 * np.eye(3),
        bias=np.zeros(3),
    )


def _make_gyro() -> Gyroscope:
    return Gyroscope(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=1e-18 * np.eye(3),
        bias=np.zeros(3),
    )


def _make_mtqs() -> list[Magnetorquer]:
    return [
        Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0),
    ]


def test_controller_inheritance_and_gain_validation() -> None:
    assert issubclass(BDot, Controller)
    assert issubclass(BCross, Controller)
    assert issubclass(NoController, Controller)
    with pytest.raises(ValueError):
        BDot(0.0)
    with pytest.raises(ValueError):
        BCross(-1.0)


def test_bdot_requires_magnetometer() -> None:
    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.1, 0.1, 0.1)), panels=[])
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(magnetorquers=_make_mtqs(), sensors=[_make_gyro()], controller=BDot(1e4)),
    )
    env = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        orbit=OrbitalState.from_cartesian((6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.5)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.5)),
        tf=1.0,
    )
    with pytest.raises(ValueError, match="Magnetometer"):
        simulate(sat, env, DynamicsConfig([], []), state, solver)


def test_bcross_requires_magnetometer_and_gyroscope() -> None:
    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.1, 0.1, 0.1)), panels=[])
    env = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        orbit=OrbitalState.from_cartesian((6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.5)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.5)),
        tf=1.0,
    )
    sat_no_gyro = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(magnetorquers=_make_mtqs(), sensors=[_make_mag()], controller=BCross(1.0)),
    )
    with pytest.raises(ValueError, match="Gyroscope"):
        simulate(sat_no_gyro, env, DynamicsConfig([], []), state, solver)


@pytest.mark.parametrize("ctrl", [BDot(5e4), BCross(0.05)])
def test_bdot_and_bcross_damp_angular_rate_in_numba_loop(ctrl: Controller) -> None:
    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.1, 0.1, 0.1)), panels=[])
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            magnetorquers=_make_mtqs(),
            sensors=[_make_mag(), _make_gyro()],
            N=1,
            controller=ctrl,
        ),
    )
    env = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    # Inclined orbit so B-field varies cleanly across axes.
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.05, -0.04, 0.03)),
        orbit=OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0),
            (0.0, 5_382.9, 5_382.9),
            2_461_000.5,
        ),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.25)),
        tf=120.0,
    )
    res = simulate(
        sat,
        env,
        DynamicsConfig([], []),
        state,
        solver,
        resultsformat=ResultsFormat(N=4, satellite_rate=True),
    )
    rates = res.satellites["sat_000"].satellite_rate
    initial_norm = np.linalg.norm(rates[0])
    final_norm = np.linalg.norm(rates[-1])
    assert final_norm < initial_norm


def test_control_and_goal_resultsformat_and_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from GenADCS.backends.python.hardware import (
        ADCSMode,
        ADCSStateMachine,
        ReactionWheel,
    )
    from GenADCS.backends.python.hardware.ADCS_subsystem.goals import NadirGoal
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
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            reaction_wheels=wheels,
            magnetorquers=_make_mtqs(),
            sensors=[_make_mag(), _make_gyro()],
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
