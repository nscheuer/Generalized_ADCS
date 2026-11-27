"""Tests for BCross controller."""

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSSubsystem,
    BCross,
    Controller,
    Magnetorquer,
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
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


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


def test_bcross_validation_and_requires_mag_and_gyro() -> None:
    assert issubclass(BCross, Controller)
    with pytest.raises(ValueError):
        BCross(-1.0)

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


def test_bcross_damps_angular_rate_in_numba_loop() -> None:
    geometry = SatelliteGeometry(mass=10.0, inertia=np.diag((0.1, 0.1, 0.1)), panels=[])
    sat = Satellite(
        geometry=geometry,
        adcs=ADCSSubsystem(
            magnetorquers=_make_mtqs(),
            sensors=[_make_mag(), _make_gyro()],
            N=1,
            controller=BCross(0.05),
        ),
    )
    env = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
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
    assert np.linalg.norm(rates[-1]) < np.linalg.norm(rates[0])
