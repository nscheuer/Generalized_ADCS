"""Behavioral tests for the hierarchical fixed-step prototype."""

import numpy as np

from .block_engine import AsyncEventEngine, FixedStepEngine, compile_fixed
from .satellite_model import PowerState, build_model


def _rates(plan):
    return {
        item.block.path: (item.requested_hz, 1.0 / (item.effective_ticks * plan.base_tick))
        for item in plan.blocks
    }


def test_nested_blocks_are_flattened_and_dead_samples_are_eliminated():
    satellite = build_model(nested_sensors=True)
    plan = compile_fixed(satellite.model)
    rates = _rates(plan)
    assert rates["sensors.gyro"] == (200.0, 10.0)
    assert rates["sensors.magnetometer"] == (20.0, 10.0)
    assert rates["sensors.packet"] == (200.0, 10.0)
    assert rates["attitude_propagation"] == (100.0, 100.0)
    assert plan.base_tick == 0.01


def test_monolithic_and_nested_sensor_models_are_numerically_equivalent():
    flat = FixedStepEngine().run(build_model(nested_sensors=False).model, 1.0, seed=8)
    nested = FixedStepEngine().run(build_model(nested_sensors=True).model, 1.0, seed=8)
    np.testing.assert_array_equal(flat.array("attitude"), nested.array("attitude"))
    flat_q = np.asarray([estimate.q for estimate in flat.history["estimate"]])
    nested_q = np.asarray([estimate.q for estimate in nested.history["estimate"]])
    np.testing.assert_array_equal(flat_q, nested_q)


def test_async_backend_uses_requested_independent_clocks():
    result = AsyncEventEngine().run(build_model(nested_sensors=True).model, 1.0, seed=2)
    assert result.execution_counts["sensors.gyro"] == 201
    assert result.execution_counts["estimator"] == 11


def test_power_subsystem_is_added_without_engine_changes():
    satellite = build_model(nested_sensors=True, with_power=True)
    result = FixedStepEngine().run(satellite.model, 1.0, seed=4)
    assert "power.solar_panels" in result.execution_counts
    assert "power.battery" in result.execution_counts
    assert all(isinstance(value, PowerState) for value in result.history["power"])
    assert all(0.0 <= value.soc <= 1.0 for value in result.history["power"])


def test_model_instance_can_be_reused_reproducibly():
    satellite = build_model(nested_sensors=True)
    first = FixedStepEngine().run(satellite.model, 0.5, seed=12)
    second = FixedStepEngine().run(satellite.model, 0.5, seed=12)
    np.testing.assert_array_equal(first.array("attitude"), second.array("attitude"))
