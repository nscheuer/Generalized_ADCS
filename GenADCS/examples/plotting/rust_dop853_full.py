"""Detailed Rust DOP853 run: four million 0.25-second intervals.

Run from the repository root with:
    venv/bin/python GenADCS/examples/plotting/rust_dop853_full.py

Diagnostics are recorded every 1,000 intervals to keep result processing small.
Plots are displayed and are not saved.
"""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "rust"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")

from importlib import import_module

from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
NRLMSISE00 = _environment.NRLMSISE00
_environment_magnetic_field_model = import_module(f"GenADCS.backends.{backed}.environment.magnetic_field_model")
IGRF14 = _environment_magnetic_field_model.IGRF14
_hardware = import_module(f"GenADCS.backends.{backed}.hardware")
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
SatellitePanel = _hardware.SatellitePanel
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
    plot, SatelliteAttitudeQuaternionPlot, BodyAngularRatePlot, PositionPlot,
    VelocityPlot, AltitudePlot, LatLonPlot, SunVectorPlot, EarthVectorPlot,
    EclipseStatePlot, AtmosphericDensityPlot, RelativeAtmosphericVelocityPlot,
    MagneticFieldPlot, MagneticFieldMagnitudePlot, AerodynamicForcePlot,
    AerodynamicTorquePlot, SRPForcePlot, SRPTorquePlot, LunisolarForcePlot,
    GravityGradientTorquePlot, JForcePlot, MagneticTorquePlot, ForcePlot, TorquePlot,
)


def main():
    dt, steps = 0.25, 40_000
    panels = [
        SatellitePanel((0.5, 0.0, 0.0), (0.0, np.pi / 2, 0.0), (0.3, 0.4)),
        SatellitePanel((-0.5, 0.0, 0.0), (0.0, -np.pi / 2, 0.0), (0.3, 0.4)),
        SatellitePanel((0.0, 0.5, 0.0), (-np.pi / 2, 0.0, 0.0), (0.4, 0.3)),
        SatellitePanel((0.0, -0.5, 0.0), (np.pi / 2, 0.0, 0.0), (0.4, 0.3)),
        SatellitePanel((0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.3, 0.3)),
        SatellitePanel((0.0, 0.0, -0.4), (np.pi, 0.0, 0.0), (0.3, 0.3)),
    ]
    geometry = SatelliteGeometry(12.0, np.diag((0.13, 0.10, 0.05)), panels=panels)
    satellite = Satellite(geometry=geometry)
    environment = Environment(planetary_body=Earth(), atmosphere=NRLMSISE00(),
                              ephemeris=DE421(), magnetic_field=IGRF14())
    dynamics = DynamicsConfig(
        forces=[
            ZonalForceModel(geometry, environment, degrees=(2, 3, 4, 5, 6)),
            ProjectedAreaDragForceModel(geometry),
            ProjectedAreaSRPForceModel(geometry),
            LunisolarForceModel(geometry, environment.ephemeris),
        ],
        torques=[
            ProjectedAreaDragTorqueModel(geometry),
            ProjectedAreaSRPTorqueModel(geometry),
            StaticGravityGradientModel(geometry),
            StaticResidualDipoleModel((0.2, -0.1, 0.15)),
        ],
    )
    state = State(
        attitude=AttitudeState((1.0, 0.0, 0.0, 0.0), (0.01, -0.015, 0.02)),
        orbit=OrbitalState.from_cartesian((6_878_137.0, 0.0, 0.0),
                                          (0.0, 7_612.6, 0.0), 2_461_000.5),
    )
    solver = CoupledSolver(CoupledDOP853Propagator(dt=dt), tf=steps * dt, progress=True)
    format = ResultsFormat(
        satellite_attitude=True, satellite_rate=True, satellite_wheel_momenta=False,
        satellite_position=True, satellite_velocity=True,
        relative_sun_vector=True, relative_earth_vector=True,
        atmospheric_density=True, relative_atmospheric_velocity=True,
        magnetic_field=True, eclipse_state=True,
        gravity_force=True, aerodynamic_force=True, solar_radiation_force=True,
        J2_force=True, J3_force=True, J4_force=True, J5_force=True, J6_force=True,
        lunisolar_force=True, gravity_gradient_torque=True,
        aerodynamic_torque=True, solar_radiation_torque=True, magnetic_torque=True,
        total_force=True, total_torque=True, N=50,
    )
    started = perf_counter()
    result = simulate(satellite, environment, dynamics, state, solver,
                      backend={"python": "Python", "cpp": "C", "rust": "Rust"}[backed], resultsformat=format)
    print(f"{backed} DOP853: {steps:,} intervals, {len(result.satellites['sat_000'].times_s):,} samples, "
          f"{perf_counter() - started:.1f} s")

    groups = [
        ("Attitude and orbit", (2, 3), [SatelliteAttitudeQuaternionPlot(), BodyAngularRatePlot(),
                                       PositionPlot(), VelocityPlot(), AltitudePlot(), LatLonPlot()]),
        ("Environment", (3, 3), [SunVectorPlot(), EarthVectorPlot(), EclipseStatePlot(),
                                  AtmosphericDensityPlot(), RelativeAtmosphericVelocityPlot(),
                                  MagneticFieldPlot(), MagneticFieldMagnitudePlot()]),
        ("Forces", (2, 3), [JForcePlot(), AerodynamicForcePlot(), SRPForcePlot(),
                             LunisolarForcePlot(), ForcePlot()]),
        ("Torques", (2, 3), [GravityGradientTorquePlot(), AerodynamicTorquePlot(),
                              SRPTorquePlot(), MagneticTorquePlot(), TorquePlot()]),
    ]
    for title, layout, panels in groups:
        plot(result, *panels, layout=layout, title=title,
             figure_kwargs={"figsize": (15, 8)}, show=False)
    plt.show()
    return result


if __name__ == "__main__":
    main()
