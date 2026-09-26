"""Shared satellite dynamics and sensor/controller case for examples 4 and 5."""

from __future__ import annotations

import numpy as np

from Z_BLOCKY import (
    Block, Dynamics, Environment, GaussianNoise, Plant, Port, Simulation,
    State, StateSampledBlock,
)


class Gyro(StateSampledBlock):
    def __init__(self) -> None:
        super().__init__(
            "gyro", state_name="attitude", frequency_hz=20.0,
            execution_s=0.004, delay_s=0.025,
            delay_noise=GaussianNoise(0.030, 0.035),
            outputs=(Port("angular_rate", float), Port("capture_time", float)),
        )

    def sample(self, state, inputs, context):
        return {"angular_rate": float(state[1]), "capture_time": context.time_s}


class TruthRate(StateSampledBlock):
    def __init__(self) -> None:
        super().__init__(
            "truth_rate", state_name="attitude", frequency_hz=100.0,
            outputs=(Port("true_rate", float),),
        )

    def sample(self, state, inputs, context):
        return {"true_rate": float(state[1])}


class GyroFromBlock(Block):
    def __init__(self) -> None:
        super().__init__(
            "gyro", frequency_hz=100.0, sample_on_demand=True,
            execution_s=0.004, delay_s=0.025,
            delay_noise=GaussianNoise(0.030, 0.035),
            inputs=(Port("true_rate", float),),
            outputs=(Port("angular_rate", float), Port("capture_time", float)),
        )

    def step(self, inputs, context=None):
        return {"angular_rate": inputs["true_rate"], "capture_time": context.time_s}


class Controller(Block):
    def __init__(self) -> None:
        super().__init__(
            "controller", frequency_hz=1.0, execution_s=0.015,
            delay_s=0.005,
            inputs=(Port("angular_rate", float), Port("capture_time", float)),
            outputs=(Port("torque", float),),
        )
        self.observations: list[tuple[float, float, float]] = []

    def step(self, inputs, context=None):
        self.observations.append((context.time_s, inputs["capture_time"], inputs["angular_rate"]))
        return {"torque": -0.7 * inputs["angular_rate"]}


def _plant_with_dynamics() -> Plant:
    plant = Plant("satellite")
    plant.add_state(State("attitude", np.ndarray, np.array([0.0, 0.3])))
    plant.set_dynamics(Dynamics(
        "attitude",
        lambda t, state, controls: np.array([
            state[1], -0.12 * state[1] + controls.get("torque", 0.0),
        ]),
        max_step_s=0.08,
    ))
    return plant


def build_case() -> tuple[Simulation, Controller]:
    plant = _plant_with_dynamics()
    plant.add(Gyro())
    controller = plant.add(Controller())
    plant.auto_connect()
    plant.bind_control(controller, "torque", "torque")
    return Simulation(environment=Environment(), plants=(plant,)), controller


def build_chained_case() -> tuple[Simulation, Controller]:
    """100 Hz state source → 100 Hz gyro → 1 Hz controller."""
    plant = _plant_with_dynamics()
    plant.add(TruthRate())
    plant.add(GyroFromBlock())
    controller = plant.add(Controller())
    plant.auto_connect()
    plant.bind_control(controller, "torque", "torque")
    return Simulation(environment=Environment(), plants=(plant,)), controller
