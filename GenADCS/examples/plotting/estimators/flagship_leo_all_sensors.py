"""High fidelity LEO run with every modeled sensor and MEKF performance.

Run from the repository root with:
    venv/bin/python GenADCS/examples/plotting/sensors/flagship_leo_all_sensors.py
Figures are grouped by spacecraft state, attitude sensors, optical sensors,
and navigation sensors. They are displayed and not saved.
"""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "cpp"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_config = import_module(f"GenADCS.backends.{backed}.config")
DynamicsConfig = _config.DynamicsConfig
_dynamics = import_module(f"GenADCS.backends.{backed}.dynamics")
ZonalForceModel = _dynamics.ZonalForceModel
ProjectedAreaDragForceModel = _dynamics.ProjectedAreaDragForceModel
ProjectedAreaDragTorqueModel = _dynamics.ProjectedAreaDragTorqueModel
ProjectedAreaSRPForceModel = _dynamics.ProjectedAreaSRPForceModel
ProjectedAreaSRPTorqueModel = _dynamics.ProjectedAreaSRPTorqueModel
LunisolarForceModel = _dynamics.LunisolarForceModel
StaticGravityGradientModel = _dynamics.StaticGravityGradientModel
StaticResidualDipoleModel = _dynamics.StaticResidualDipoleModel
_environment = import_module(f"GenADCS.backends.{backed}.environment")
DE421 = _environment.DE421
Earth = _environment.Earth
Environment = _environment.Environment
IGRF14 = import_module(f"GenADCS.backends.{backed}.environment.magnetic_field_model").IGRF14
NRLMSISE00 = _environment.NRLMSISE00
_hardware = import_module(f"GenADCS.backends.{backed}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
SatellitePanel = _hardware.SatellitePanel
_hardware_ADCS_subsystem = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem")
ADCSSubsystem = _hardware_ADCS_subsystem.ADCSSubsystem
_hardware_ADCS_subsystem_sensors = import_module(f"GenADCS.backends.{backed}.hardware.ADCS_subsystem.sensors")
Gyroscope = _hardware_ADCS_subsystem_sensors.Gyroscope
Magnetometer = _hardware_ADCS_subsystem_sensors.Magnetometer
AutonomousStarTracker = _hardware_ADCS_subsystem_sensors.AutonomousStarTracker
CoarseSunSensor = _hardware_ADCS_subsystem_sensors.CoarseSunSensor
FineSunSensor = _hardware_ADCS_subsystem_sensors.FineSunSensor
FineEarthSensor = _hardware_ADCS_subsystem_sensors.FineEarthSensor
GNSS = _hardware_ADCS_subsystem_sensors.GNSS
Accelerometer = _hardware_ADCS_subsystem_sensors.Accelerometer
AttitudeMEKF = _hardware_ADCS_subsystem.AttitudeMEKF
_propagation = import_module(f"GenADCS.backends.{backed}.propagation")
CoupledDOP853Propagator = _propagation.CoupledDOP853Propagator
_simulate = import_module(f"GenADCS.backends.{backed}.simulate")
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
_solver = import_module(f"GenADCS.backends.{backed}.solver")
CoupledSolver = _solver.CoupledSolver
_state = import_module(f"GenADCS.backends.{backed}.state")
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
from GenADCS.backends.shared.plotting import (
    plot, SatelliteAttitudeQuaternionPlot, BodyAngularRatePlot,
    GyroscopePlot, MagnetometerPlot, StarTrackerPlot, CoarseSunSensorPlot,
    FineSunSensorPlot, FineEarthSensorPlot, GNSSPositionPlot,
    GNSSVelocityPlot, AccelerometerPlot,
    AbsoluteKnowledgeErrorQuaternionPlot, AbsoluteKnowledgeErrorRatePlot,
    EstimatedAttitudeCovariancePlot, EstimatedRateCovariancePlot,
    NormalizedInnovationSquaredPlot, NormalizedEstimationErrorSquaredPlot,
    MeasurementInnovationCovariancePlot, MeasurementResidualPlot,
)


def _q_identity():
    return np.array([1., 0., 0., 0.])


def _mount(axis):
    """Quaternion rotating the selected body face normal (+Z) onto axis."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    z = np.array([0., 0., 1.])
    dot = np.clip(np.dot(z, axis), -1., 1.)
    if dot < -0.999999:
        return np.array([0., 1., 0., 0.])
    cross = np.cross(z, axis)
    q = np.r_[1. + dot, cross]
    return q / np.linalg.norm(q)


def _sensor_noise(sigmas):
    return np.diag(np.square(np.asarray(sigmas, dtype=float)))


def main():
    dt, duration_s, record_every = 0.5, 10000., 4
    identity = _q_identity()
    panels = [
        SatellitePanel((0.85, 0., 0.), (0., np.pi/2, 0.), (1.8, 0.8), .25, .25, .50),
        SatellitePanel((-0.85, 0., 0.), (0., -np.pi/2, 0.), (1.8, 0.8), .25, .25, .50),
        SatellitePanel((0., 0.85, 0.), (-np.pi/2, 0., 0.), (1.8, 0.8), .25, .25, .50),
        SatellitePanel((0., -0.85, 0.), (np.pi/2, 0., 0.), (1.8, 0.8), .25, .25, .50),
        SatellitePanel((0., 0., 0.65), (0., 0., 0.), (1.2, 1.2), .25, .25, .50),
        SatellitePanel((0., 0., -0.65), (np.pi, 0., 0.), (1.2, 1.2), .25, .25, .50),
    ]
    geometry = SatelliteGeometry(
        mass=620., inertia=np.diag((520., 470., 310.)), panels=panels,
        center_of_mass=(0., 0., 0.),
    )
    environment = Environment(planetary_body=Earth(), atmosphere=NRLMSISE00(),
                              ephemeris=DE421(), magnetic_field=IGRF14())

    sensors = [
        Gyroscope(np.zeros(3), identity, identity, 0.2,
                  _sensor_noise([2e-5]*3), np.array([1.2e-5, -0.8e-5, 0.5e-5])),
        Magnetometer(np.array([.35, 0., 0.]), identity, identity, 8e-4,
                     _sensor_noise([80e-9]*3), np.array([12e-9, -8e-9, 5e-9])),
        AutonomousStarTracker(np.array([0., 0., .4]), identity, identity, 1.,
                              _sensor_noise([8e-6]*3), np.zeros(3),
                              fov_half_angle=np.deg2rad(25.)),
    ]
    # Six body-face photodiodes, two fine Sun heads, one nadir Earth head.
    for axis in ([1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]):
        sensors.append(CoarseSunSensor(np.asarray(axis)*.9, _mount(axis), identity,
            np.array([0., 0., 1.]), 1.0, 1.0, 2.5e-5, 0.0, np.deg2rad(80.)))
    sensors.extend([
        FineSunSensor(np.array([.4, 0., .2]), identity, identity,
                      _sensor_noise([np.deg2rad(0.003)]*2), np.zeros(2),
                      np.deg2rad([35., 35.])),
        FineSunSensor(np.array([-.4, 0., .2]), identity, identity,
                      _sensor_noise([np.deg2rad(0.003)]*2), np.zeros(2),
                      np.deg2rad([35., 35.])),
        FineEarthSensor(np.array([0., 0., -.4]), _mount([0.,0.,-1.]), identity,
                        _sensor_noise([np.deg2rad(0.005)]*2), np.zeros(2),
                        np.deg2rad([30., 30.])),
        GNSS(_sensor_noise([2., 2., 3., .015, .015, .02]),
             np.array([.4, -.3, .2, .002, -.001, .0015])),
        Accelerometer(np.array([.3, .2, -.1]), identity, identity, 2.,
                      _sensor_noise([100e-6]*3), np.array([20e-6, -15e-6, 10e-6])),
    ])
    estimator = AttitudeMEKF(
        (0.9998, 0.01, -0.015, 0.02), (0.0022, -0.0017, 0.0018),
        np.diag([np.deg2rad(2.)**2]*3 + [np.deg2rad(0.02)**2]*3),
        np.eye(3) * 1e-10,
    )
    satellite = Satellite(geometry=geometry,
                          adcs=ADCSSubsystem(sensors=sensors, estimator=estimator, N=2))
    dynamics = DynamicsConfig(
        forces=[ZonalForceModel(geometry, environment, degrees=(2,3,4,5,6)),
                ProjectedAreaDragForceModel(geometry), ProjectedAreaSRPForceModel(geometry),
                LunisolarForceModel(geometry, environment.ephemeris)],
        torques=[ProjectedAreaDragTorqueModel(geometry), ProjectedAreaSRPTorqueModel(geometry),
                 StaticGravityGradientModel(geometry),
                 StaticResidualDipoleModel((0.08, -0.05, 0.06))],
    )
    state = State(
        AttitudeState((1., 0., 0., 0.), (0.002, -0.0015, 0.001)),
        OrbitalState.from_cartesian((6_978_137., 0., 0.), (0., 7_504.3, 0.), 2_461_000.5),
    )
    formats = dict(gyroscope_clean=True, gyroscope_reading=True,
                   magnetometer_clean=True, magnetometer_reading=True,
                   autonomous_star_tracker_clean=True, autonomous_star_tracker_reading=True,
                   coarse_sun_sensor_clean=True, coarse_sun_sensor_reading=True,
                   fine_sun_sensor_clean=True, fine_sun_sensor_reading=True,
                   fine_earth_sensor_clean=True, fine_earth_sensor_reading=True,
                   gnss_clean=True, gnss_reading=True,
                   accelerometer_clean=True, accelerometer_reading=True,
                   satellite_position=True, satellite_velocity=True,
                   relative_sun_vector=True, relative_earth_vector=True,
                   eclipse_state=True, atmospheric_density=True,
                   magnetic_field=True, satellite_attitude=True, satellite_rate=True,
                   estimated_attitude=True, estimated_rate=True,
                   estimated_attitude_covariance=True, estimated_rate_covariance=True,
                   normalized_innovation_squared=True,
                   normalized_estimation_error_squared=True,
                   measurement_innovation_covariance=True,
                   measurement_residual=True, kalman_gain=True, N=record_every)
    solver = CoupledSolver(CoupledDOP853Propagator(dt=dt, rtol=2e-9, atol=1e-10,
                                                   max_step=2., min_step=1e-5),
                           tf=duration_s, progress=True, progress_chunk_steps=2000)
    started = perf_counter()
    result = simulate(satellite, environment, dynamics, state, solver,
                      backend={"python": "Python", "cpp": "C", "rust": "Rust"}[backed], resultsformat=ResultsFormat(**formats))
    sample = result.satellites["sat_000"]
    print(f"Python DOP853: {duration_s:g} s, {len(sample.times_s):,} recorded samples, "
          f"{len(sensors)} sensor units, {perf_counter()-started:.1f} s")

    groups = [
        ("Spacecraft state", (2, 2), [SatelliteAttitudeQuaternionPlot(),
                                       BodyAngularRatePlot()]),
        ("Rate and magnetic sensors", (2, 2), [GyroscopePlot(), MagnetometerPlot()]),
        ("Optical attitude sensors", (2, 2), [StarTrackerPlot(), CoarseSunSensorPlot(),
                                               FineSunSensorPlot(), FineEarthSensorPlot()]),
        ("Navigation sensors", (2, 2), [GNSSPositionPlot(), GNSSVelocityPlot(),
                                         AccelerometerPlot()]),
    ]
    groups.extend([
        ("Attitude estimate performance", (2, 2), [
            AbsoluteKnowledgeErrorQuaternionPlot(option="magnitude"),
            AbsoluteKnowledgeErrorRatePlot(option="magnitude"),
            EstimatedAttitudeCovariancePlot(option="trace"),
            EstimatedRateCovariancePlot(option="trace"),
        ]),
        ("MEKF diagnostics", (2, 2), [
            NormalizedInnovationSquaredPlot(),
            NormalizedEstimationErrorSquaredPlot(),
            MeasurementInnovationCovariancePlot(option="trace"),
            MeasurementResidualPlot(option="trace"),
        ]),
    ])
    for title, layout, panels in groups:
        plot(result, *panels, layout=layout, title=title,
             figure_kwargs={"figsize": (14, 8)}, show=False)
    plt.show()
    return result


if __name__ == "__main__":
    main()
