"""End-to-end exercise of the C++ coupled solver through Python bindings.

Run from the repository root with ``.venv/bin/python debug/debug_OMNI/debug_coupled_solver_C.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADCS_Omni.ADCS_C.state import AttitudeState, State, OrbitalState
from ADCS_Omni.ADCS_C.hardware import Satellite, SatelliteGeometry, SatellitePanel
from ADCS_Omni.ADCS_C.environment import Environment, Earth, DE421, NRLMSISE00
from ADCS_Omni.ADCS_C.config import DynamicsConfig
from ADCS_Omni.ADCS_C.propagation import CoupledRK4Propagator, CoupledDOP853Propagator
from ADCS_Omni.ADCS_C.solver import CoupledSolver
from ADCS_Omni.ADCS_C.simulate import simulate
from ADCS_Omni.ADCS_C.dynamics import (  # noqa: E402
    ProjectedAreaDragForceModel,
    ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel,
    ProjectedAreaSRPTorqueModel,
    StaticGravityGradientModel,
    ZonalForceModel,
)



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the C++ coupled solver")
    parser.add_argument("--duration", type=float, default=40000.0)
    parser.add_argument("--dt", type=float, default=0.25)
    args = parser.parse_args()
    dt = args.dt
    duration = args.duration
    steps = int(duration / dt)

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.13, 0.10, 0.05)),
        panels=[
            SatellitePanel((0.0, 0.0, 0.15), (0.0, 0.0, 0.0), (0.10, 0.20), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, 0.0, -0.15), (np.pi, 0.0, 0.0), (0.10, 0.20), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.05, 0.0, 0.0), (0.0, np.pi / 2.0, 0.0), (0.20, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((-0.05, 0.0, 0.0), (0.0, -np.pi / 2.0, 0.0), (0.20, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, 0.10, 0.0), (-np.pi / 2.0, 0.0, 0.0), (0.10, 0.30), 0.20, 0.70, 0.10, 2.2),
            SatellitePanel((0.0, -0.10, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.10, 0.30), 0.20, 0.70, 0.10, 2.2),
        ],
    )
    satellite = Satellite(geometry=geometry)
    environment = Environment(
        planetary_body=Earth(),
        atmosphere=NRLMSISE00(),
        ephemeris=DE421(),
    )
    dynamics = DynamicsConfig(
        forces=[
            ZonalForceModel(geometry, environment, degrees=(2,)),
            ProjectedAreaDragForceModel(geometry),
            ProjectedAreaSRPForceModel(geometry),
        ],
        torques=[
            StaticGravityGradientModel(geometry),
            ProjectedAreaDragTorqueModel(geometry),
            ProjectedAreaSRPTorqueModel(geometry),
        ],
    )

    state = State(
        attitude=AttitudeState(
            orientation=(0.9238795325, 0.0, 0.3826834324, 0.0),
            rates=(0.01, -0.015, 0.02),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
        ),
    )

    solver = CoupledSolver(CoupledDOP853Propagator(dt=dt), tf=duration, progress=True)
    result = simulate(satellite, environment, dynamics, state, solver, backend="C")
    state = result.final_state

    altitude = np.linalg.norm(state.orbit.position) - environment.planetary_body.reference_radius
    print(f"Elapsed: {duration:.1f} s ({steps} steps)")
    print(f"Final altitude: {altitude / 1_000:.3f} km")
    print(f"Final position [m]: {state.orbit.position}")
    print(f"Final velocity [m/s]: {state.orbit.velocity}")
    print(f"Final quaternion: {state.attitude.quaternion_array}")
    print(f"Final body rates [rad/s]: {state.attitude.rates}")


if __name__ == "__main__":
    main()
