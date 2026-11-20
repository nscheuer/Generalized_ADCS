"""Small end-to-end exercise of the Numba coupled RK4 solver.

Run from the repository root with ``.venv/bin/python debug/debug_OMNI/debug_coupled_solver.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADCS_Omni import (  # noqa: E402
    AttitudeState,
    CoupledRK4Propagator,
    CoupledDOP853Propagator,
    CoupledSolver,
    DynamicsConfig,
    DE421,
    Earth,
    Environment,
    NRLMSISE00,
    Satellite,
    SatelliteGeometry,
    SatellitePanel,
    State,
    OrbitalState,
    simulate,
)
from ADCS_Omni.ADCS_Python.dynamics import (  # noqa: E402
    ProjectedAreaDragForceModel,
    ProjectedAreaDragTorqueModel,
    ProjectedAreaSRPForceModel,
    ProjectedAreaSRPTorqueModel,
    StaticGravityGradientModel,
    ZonalForceModel,
)


def main():
    dt = 0.25
    duration = 40000.0
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
    result = simulate(satellite, environment, dynamics, state, solver)
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
