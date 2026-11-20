"""Small end-to-end exercise of the Numba coupled RK4 solver.

Run from the repository root with ``.venv/bin/python debug/debug_OMNI/debug_coupled_solver.py``.
"""

from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADCS_Omni import (  # noqa: E402
    AttitudeState,
    CoupledPropagator,
    DynamicsConfig,
    Earth,
    Environment,
    Satellite,
    SatelliteGeometry,
    State,
    OrbitalState,
)
from ADCS_Omni.dynamics.gravity_gradient_model import StaticGravityGradientModel  # noqa: E402


def main():
    dt = 0.25
    duration = 6000.0
    steps = int(duration / dt)

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.13, 0.10, 0.05)),
    )
    satellite = Satellite(geometry=geometry)
    environment = Environment(planetary_body=Earth())
    dynamics = DynamicsConfig(
        gravity_gradient_torque=True,
        gravity_gradient_model=StaticGravityGradientModel(geometry),
        j2_force=True,
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

    solver = CoupledPropagator(dt=dt)
    # Compile the Numba specialization with a disposable step so the run bar
    # measures the requested simulation only.
    with tqdm(total=1, desc="Compiling Numba kernels", unit="step") as bar:
        solver.step(state, satellite, environment, dynamics)
        bar.update(1)

    for _ in tqdm(range(steps), desc="Running coupled RK4", unit="step"):
        state = solver.step(state, satellite, environment, dynamics)

    altitude = np.linalg.norm(state.orbit.position) - environment.planetary_body.reference_radius
    print(f"Elapsed: {duration:.1f} s ({steps} steps)")
    print(f"Final altitude: {altitude / 1_000:.3f} km")
    print(f"Final position [m]: {state.orbit.position}")
    print(f"Final velocity [m/s]: {state.orbit.velocity}")
    print(f"Final quaternion: {state.attitude.quaternion_array}")
    print(f"Final body rates [rad/s]: {state.attitude.rates}")


if __name__ == "__main__":
    main()
