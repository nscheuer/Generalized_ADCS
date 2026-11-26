import numpy as np
import pytest

from GenADCS.backends.python import simulate
from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    quaternion_to_dcm,
)
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    ReferenceAttitudePropagator,
)
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_reference_attitude_propagator_modes_and_validation() -> None:
    nadir = ReferenceAttitudePropagator(0.5, "nadir")
    assert nadir.dt == 0.5
    assert nadir.mode == 0

    along = ReferenceAttitudePropagator(1.0, "along-track")
    assert along.mode == 1

    sun = ReferenceAttitudePropagator(1.0, "sun")
    assert sun.mode == 2

    with pytest.raises(ValueError, match="dt must be finite and positive"):
        ReferenceAttitudePropagator(0.0, "nadir")
    with pytest.raises(ValueError, match="unknown reference attitude mode"):
        ReferenceAttitudePropagator(0.5, "invalid")


def test_reference_attitude_nadir_and_along_track_orientation() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = np.sqrt(mu / radius)
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        OrbitalState.from_cartesian([radius, 0.0, 0.0], [0.0, speed, 0.0], 2451545.0),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))

    nadir_result = simulate(
        satellite,
        environment,
        DynamicsConfig(),
        state,
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
            attitude=AttitudeSolver(ReferenceAttitudePropagator(0.25, "nadir")),
            tf=0.5,
        ),
    )
    nadir_q = nadir_result.satellites["sat_000"].fields["satellite_attitude"][0]
    nadir_dcm = quaternion_to_dcm(nadir_q)
    np.testing.assert_allclose(nadir_dcm[2], [-1.0, 0.0, 0.0], atol=1e-12)

    along_result = simulate(
        satellite,
        environment,
        DynamicsConfig(),
        state,
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(0.25)),
            attitude=AttitudeSolver(ReferenceAttitudePropagator(0.25, "along-track")),
            tf=0.5,
        ),
    )
    along_q = along_result.satellites["sat_000"].fields["satellite_attitude"][0]
    along_dcm = quaternion_to_dcm(along_q)
    np.testing.assert_allclose(along_dcm[2], [0.0, 1.0, 0.0], atol=1e-12)
