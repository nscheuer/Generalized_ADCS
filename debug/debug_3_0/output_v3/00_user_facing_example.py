from debug.debug_3_0.user_api import (
    Environment, InitialState, Integration,
    Scenario, Satellite, Simulation,
)

# Satellite owns its hardware, flight software, and initial state.
satellite = Satellite.standard_cube_sat(
    name="solo",
    sensors=False,
    estimator=False,     # controller receives ideal state knowledge
    initial_state=InitialState.leo_default(),
)

environment = Environment.standard_leo(
    gravity="J2",
    magnetic_field="dipole",
    atmosphere="exponential",
)

# Scenario owns numerical and execution policy.
scenario = Scenario(
    duration=60.0,
    kernel="Numba",     # "Python", "Numba", or "JAX"
    run="Single",       # "Single" or "Monte Carlo"
    integration=Integration(
        orbit_hz=1.0,
        attitude_hz=1.0,
        subsystem_hz=1.0,
    ),
)

simulation = Simulation(
    satellites={"solo": satellite},
    environment=environment,
    scenario=scenario,
)

result = simulation.run()
