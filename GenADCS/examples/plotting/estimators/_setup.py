"""Common observable MEKF simulation for estimator plots."""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "python"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module

import numpy as np


def run_example(progress=True, backed=backed):
    if backed not in ("rust", "cpp", "python"):
        raise ValueError("backed must be rust, cpp, or python")
    _config = import_module(f"GenADCS.backends.{backed}.config")
    DynamicsConfig = _config.DynamicsConfig
    _environment = import_module(f"GenADCS.backends.{backed}.environment")
    DE421 = _environment.DE421
    Earth = _environment.Earth
    Environment = _environment.Environment
    _hardware = import_module(f"GenADCS.backends.{backed}.hardware")
    Satellite = _hardware.Satellite
    SatelliteGeometry = _hardware.SatelliteGeometry
    _hardware_ADCS_subsystem = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem")
    ADCSSubsystem = _hardware_ADCS_subsystem.ADCSSubsystem
    AttitudeMEKF = _hardware_ADCS_subsystem.AttitudeMEKF
    _hardware_ADCS_subsystem_sensors = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem.sensors")
    AutonomousStarTracker = _hardware_ADCS_subsystem_sensors.AutonomousStarTracker
    Gyroscope = _hardware_ADCS_subsystem_sensors.Gyroscope
    _propagation = import_module(f"GenADCS.backends.{backed}.propagation")
    CoupledRK4Propagator = _propagation.CoupledRK4Propagator
    _simulate = import_module(f"GenADCS.backends.{backed}.simulate")
    ResultsFormat = _simulate.ResultsFormat
    simulate = _simulate.simulate
    _solver = import_module(f"GenADCS.backends.{backed}.solver")
    CoupledSolver = _solver.CoupledSolver
    _state = import_module(f"GenADCS.backends.{backed}.state")
    AttitudeState = _state.AttitudeState
    OrbitalState = _state.OrbitalState
    State = _state.State

    identity = np.array([1., 0., 0., 0.])
    rate = np.array([.01, -.015, .02])
    gyro = Gyroscope(np.zeros(3), identity, identity, 10., np.eye(3) * 1e-8, np.zeros(3))
    tracker = AutonomousStarTracker(np.zeros(3), identity, identity, 1.,
                                    np.eye(3) * 1e-6, np.zeros(3),
                                    fov_half_angle=np.deg2rad(5.))
    initial_attitude = np.array([np.cos(.1), np.sin(.1), 0., 0.])
    estimator = AttitudeMEKF(initial_attitude, rate + np.array([1e-3, -1e-3, 2e-3]),
                             np.eye(6) * 1e-2, 1e-8)
    satellite = Satellite(SatelliteGeometry(12., np.diag([.13, .1, .05]), []),
                          ADCSSubsystem(sensors=[gyro, tracker], estimator=estimator))
    state = State(AttitudeState(identity, rate),
                  OrbitalState.from_cartesian((6878137., 0., 0.), (0., 7612.6, 0.), 2461000.5))
    return simulate(satellite, Environment(planetary_body=Earth(), ephemeris=DE421()),
                    DynamicsConfig(), state, CoupledSolver(CoupledRK4Propagator(.25),
                    tf=120., progress=progress), resultsformat=ResultsFormat(
                    satellite_attitude=True, satellite_rate=True, estimated_attitude=True,
                    estimated_rate=True, estimated_attitude_covariance=True,
                    estimated_rate_covariance=True, measurement_innovation=True,
                    measurement_innovation_covariance=True,
                    normalized_innovation_squared=True,
                    normalized_estimation_error_squared=True,
                    measurement_residual=True, kalman_gain=True))
