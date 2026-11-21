"""Four-wheel coupled orbit and attitude propagation using ADCS wheel models.

Run from the repository root with ``.venv/bin/python debug/debug_OMNI/debug_coupled_solver_wheel.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADCS_Omni import (  # noqa: E402
    ADCSSubsystem,
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
    WheelMomentumState,
    OrbitalState,
    simulate,
)
from ADCS.satellite_hardware.actuators.reaction_wheel import RW  # noqa: E402
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
    duration = 400000.0
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
    # Tetrahedral four-wheel cluster. ADCS.RW uses scalar h and a body torque
    # command about each body-frame axis; the compiled model uses the same
    # torque/storage sign and rotor-inertia convention.
    axes = np.array(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=float) / np.sqrt(3.0)
    adcs = ADCSSubsystem(reaction_wheels=[
        RW(axis=axis, max_torque=2e-5, J=2e-4, h=h, h_max=0.02)
        for axis, h in zip(axes, (0.001, -0.001, 0.0005, -0.0005))
    ])
    satellite = Satellite(geometry=geometry, adcs=adcs)
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
        internal=WheelMomentumState(adcs),
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
    print(f"Final wheel momenta [N m s]: {state.internal.wheel_momentum}")


if __name__ == "__main__":
    main()
