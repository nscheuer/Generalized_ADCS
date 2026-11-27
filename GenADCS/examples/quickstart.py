from importlib import import_module

import numpy as np

backend = "python"  # or "cpp" / "rust" once those backends are built
B = lambda module: import_module(f"GenADCS.backends.{backend}.{module}")
state, hardware, env = B("state"), B("hardware"), B("environment")

# A 12 kg spacecraft tumbling on a 500 km orbit with J2 gravity
geometry = hardware.SatelliteGeometry(mass=12.0, inertia=np.diag((0.13, 0.10, 0.05)), panels=[])
satellite = hardware.Satellite(geometry=geometry)
environment = env.Environment(planetary_body=env.Earth())
dynamics = B("config").DynamicsConfig(
    forces=[B("dynamics").ZonalForceModel(geometry, environment, degrees=(2,))], torques=[])
initial = state.State(
    attitude=state.AttitudeState(orientation=(1.0, 0.0, 0.0, 0.0), rates=(0.01, -0.015, 0.02)),
    orbit=state.OrbitalState.from_cartesian(position=(6_878_137.0, 0.0, 0.0),
                                            velocity=(0.0, 7_612.6, 0.0), julian_date=2_461_000.5),
)
solver = B("solver").CoupledSolver(B("propagation").CoupledRK4Propagator(dt=0.25), tf=5_677.0)

result = B("simulate").simulate(satellite, environment, dynamics, initial, solver,
                                backend={"python": "Python", "cpp": "C", "rust": "Rust"}[backend])
run = result.satellites["sat_000"]
print(f"{backend}: {len(run.times_s):,} samples, final attitude {np.round(run.fields['satellite_attitude'][-1], 6)}")
