import numpy as np
import pytest

from GenADCS.backends.python import simulate
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import AttitudeMEKF
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope
from GenADCS.backends.python.helpers.input_output.resultsformat import ResultsFormat
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.propagation.separate_rk4 import _integrate_separate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_simulate_uses_one_compiled_loop_for_separate_propagators() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian(
            [radius, 0.0, 0.0], [0.0, np.sqrt(mu / radius), 0.0], 2451545.0
        ),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
        attitude=AttitudeSolver(RK4AttitudePropagator(0.25)),
        tf=1.1,
    )
    result = simulate(
        satellite,
        environment,
        DynamicsConfig(),
        state,
        solver,
        resultsformat=ResultsFormat(satellite_position=True, satellite_velocity=True),
    )
    samples = result.satellites["sat_000"]
    np.testing.assert_allclose(samples.times_s, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        samples.fields["satellite_attitude"][-1],
        [np.cos(0.05), 0.0, 0.0, -np.sin(0.05)],
        atol=1e-8,
    )
    assert samples.fields["satellite_position"][-1, 1] > 0
    assert _integrate_separate.nopython_signatures


def test_separate_solver_requires_matching_step_sizes() -> None:
    with pytest.raises(ValueError, match="matching dt"):
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(1.0)),
            attitude=AttitudeSolver(RK4AttitudePropagator(0.5)),
            tf=10.0,
        )


def test_separate_simulation_samples_sensors_inside_compiled_loop() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = Gyroscope(
        np.zeros(3),
        identity,
        identity,
        10.0,
        np.eye(3) * 1e-12,
        np.zeros(3),
    )
    satellite = Satellite(
        SatelliteGeometry(10.0, np.eye(3)),
        ADCSSubsystem(sensors=[gyro], N=2),
    )
    state = State(
        AttitudeState(identity, np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
    )
    result = simulate(
        satellite,
        Environment(),
        DynamicsConfig(),
        state,
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
            attitude=AttitudeSolver(RK4AttitudePropagator(0.25)),
            tf=0.75,
        ),
        resultsformat=ResultsFormat(gyroscope_reading=True),
        seed=7,
    )
    readings = result.satellites["sat_000"].fields["gyroscope_reading"]
    assert np.isfinite(readings[0]).all()
    assert np.isnan(readings[1]).all()
    assert np.isfinite(readings[2]).all()


def test_separate_simulation_updates_estimator_inside_compiled_loop() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = Gyroscope(
        np.zeros(3),
        identity,
        identity,
        10.0,
        np.eye(3) * 1e-8,
        np.zeros(3),
    )
    estimator = AttitudeMEKF(identity, np.array([0.1, 0.1, 0.1]), np.eye(6) * 0.1, 1e-8)
    satellite = Satellite(
        SatelliteGeometry(10.0, np.eye(3)),
        ADCSSubsystem(sensors=[gyro], estimator=estimator),
    )
    state = State(
        AttitudeState(identity, np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
    )
    result = simulate(
        satellite,
        Environment(),
        DynamicsConfig(),
        state,
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
            attitude=AttitudeSolver(RK4AttitudePropagator(0.25)),
            tf=0.5,
        ),
        resultsformat=ResultsFormat(estimated_attitude=True, estimated_rate=True),
    )
    fields = result.satellites["sat_000"].fields
    assert np.isfinite(fields["estimated_attitude"]).all()
    assert np.isfinite(fields["estimated_rate"]).all()


def test_benchmark_progress_bar_covers_all_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from GenADCS.examples.benchmarks.keplerian_solver_rk4 import main

    result = main(steps=8)
    assert len(result.satellites["sat_000"].times_s) == 9
    progress_output = capsys.readouterr().err
    assert "Running Keplerian + RK4" in progress_output
    assert "8/8" in progress_output
