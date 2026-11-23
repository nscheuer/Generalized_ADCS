"""Detailed C++ DOP853 run: four million 0.25-second intervals.

Run from the repository root with:
    venv/bin/python GenADCS/examples/plotting/cpp_dop853_full.py

Diagnostics are recorded every 1,000 intervals to keep result processing small.
Plots are displayed and are not saved.
"""

from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from GenADCS.backends.cpp.config import DynamicsConfig
from GenADCS.backends.cpp.dynamics import (
    ZonalForceModel, ProjectedAreaDragForceModel, ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel, ProjectedAreaSRPTorqueModel,
    LunisolarForceModel, StaticGravityGradientModel, StaticResidualDipoleModel,
)
from GenADCS.backends.cpp.environment import DE421, Earth, Environment, IGRF14, NRLMSISE00
from GenADCS.backends.cpp.hardware import Satellite, SatelliteGeometry, SatellitePanel
from GenADCS.backends.cpp.propagation import CoupledDOP853Propagator
from GenADCS.backends.cpp.simulate import ResultsFormat, simulate
from GenADCS.backends.cpp.solver import CoupledSolver
from GenADCS.backends.cpp.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.plotting import (
    plot, SatelliteAttitudeQuaternionPlot, BodyAngularRatePlot, PositionPlot,
    VelocityPlot, AltitudePlot, LatLonPlot, SunVectorPlot, EarthVectorPlot,
    EclipseStatePlot, AtmosphericDensityPlot, RelativeAtmosphericVelocityPlot,
    MagneticFieldPlot, MagneticFieldMagnitudePlot, AerodynamicForcePlot,
    AerodynamicTorquePlot, SRPForcePlot, SRPTorquePlot, LunisolarForcePlot,
    GravityGradientTorquePlot, JForcePlot, MagneticTorquePlot, ForcePlot, TorquePlot,
)


def main():
    dt, steps = 0.25, 4_000_000
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
        total_force=True, total_torque=True, N=1000,
    )
    started = perf_counter()
    result = simulate(satellite, environment, dynamics, state, solver,
                      backend="C", resultsformat=format)
    print(f"C++ DOP853: {steps:,} intervals, {len(result.satellites['sat_000'].times_s):,} samples, "
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
