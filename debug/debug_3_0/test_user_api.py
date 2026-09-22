"""Checks for the v3 public composition layer."""

import numpy as np

from .user_api import Environment, InitialState, Integration, Scenario, Satellite, Simulation


def _single(*, sensors=False, estimator=False, kernel="Python"):
    satellite = Satellite.standard_cube_sat(
        "solo", sensors=sensors, estimator=estimator, initial_state=InitialState.leo_default(),
    )
    return Simulation(
        satellites={"solo": satellite}, environment=Environment.standard_leo(),
        scenario=Scenario(
            duration=2.0, kernel=kernel, run="Single",
            integration=Integration(orbit_hz=1, attitude_hz=1, subsystem_hz=1, orbit_integrator="rk2", attitude_integrator="rk2"),
        ),
    )


def test_satellite_owns_initial_state_and_no_sensor_path_is_explicit():
    simulation = _single()
    assert simulation.satellites["solo"].initial_state.orbit.shape == (6,)
    paths = {block.path for block in simulation.compile().model.leaves()}
    assert "solo.state_knowledge" in paths
    assert "solo.sensors" not in paths
    assert "solo.estimator" not in paths


def test_scenario_integrators_configure_runtime_propagators():
    compiled = _single().compile()
    leaves = {block.path: block for block in compiled.model.leaves()}
    assert leaves["solo.orbit_propagation"].integrator == "rk2"
    assert leaves["solo.attitude_propagation"].order == 2


def test_numba_single_uses_public_scenario_kernel():
    result = _single(kernel="Numba").run()
    assert result.backend == "numba-fixed-complete"
    assert result.orbit is not None
    assert result.estimate is not None
    assert result.attitude.shape[1] == 1


def test_two_satellites_get_independent_state_and_shared_formation_block():
    chief_state = InitialState.leo_default()
    chief = Satellite.standard_cube_sat("chief", sensors=False, estimator=False, initial_state=chief_state)
    deputy_orbit = chief_state.orbit.copy(); deputy_orbit[0] += 100.0
    deputy = Satellite.standard_cube_sat(
        "deputy", sensors=False, estimator=False,
        initial_state=InitialState(deputy_orbit, chief_state.attitude.copy()),
    )
    simulation = Simulation(
        satellites={"chief": chief, "deputy": deputy}, environment=Environment.standard_leo(),
        scenario=Scenario(
            duration=2.0,
            integration=Integration(orbit_hz=1, attitude_hz=1, subsystem_hz=1),
        ),
    )
    paths = {block.path for block in simulation.compile().model.leaves()}
    assert {"chief.gps", "deputy.gps", "formation_controller"} <= paths
    result = simulation.run()
    assert np.asarray(result.history["relative_position_m"]).shape[1] == 3
