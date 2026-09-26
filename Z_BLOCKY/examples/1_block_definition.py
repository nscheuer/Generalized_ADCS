"""Define an event block and a queried environmental model."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Z_BLOCKY import (
    Block, Environment, GaussianNoise, Plant, Port, QueriedBlock,
    QueryPort, Simulation, State,
)


class Disturbance(QueriedBlock):
    """Pure model queried at the time/state requested by its consumer."""

    def __init__(self) -> None:
        super().__init__(
            "disturbance",
            queries=(QueryPort("torque", request_type=dict, response_type=float),),
            delay_s=0.002,
            delay_noise=GaussianNoise(standard_deviation=0.0001),
        )

    def query(self, port, request, context=None):
        torque = 0.03 * request["attitude_error"]
        print(f"Disturbance.query({port}) -> torque={torque:.4f} Nm")
        return torque


class Estimator(Block):
    """Runs when a sensor event arrives; it has no periodic trigger."""

    def __init__(self) -> None:
        super().__init__(
            "estimator",
            frequency_hz=20.0,  # nominal maximum/expected event rate metadata
            delay_s=0.010,
            delay_noise=GaussianNoise(standard_deviation=0.0005),
            inputs=(Port("gyro_measurement", float),),
            outputs=(Port("estimated_rate", float),),
        )

    def step(self, inputs, context=None):
        rate = inputs["gyro_measurement"]
        estimate = 0.9 * rate
        print(f"Estimator.step() <- gyro={rate:.3f} rad/s; -> estimate={estimate:.3f}")
        return {"estimated_rate": estimate}


class Gyroscope(Block):
    """Sensor model emits events at its configured nominal event rate."""

    def __init__(self) -> None:
        super().__init__(
            "gyroscope",
            frequency_hz=20.0,
            outputs=(Port("gyro_measurement", float),),
        )

    def step(self, inputs, context=None):
        measurement = 0.4
        print(f"Gyroscope event -> gyro={measurement:.3f} rad/s")
        return {"gyro_measurement": measurement}


class Controller(Block):
    def __init__(self) -> None:
        super().__init__(
            "controller",
            frequency_hz=1.0,
            inputs=(Port("estimated_rate", float),),
            outputs=(Port("actuator_command", float),),
        )

    def step(self, inputs, context=None):
        command = -inputs["estimated_rate"]
        print(f"Controller.step() <- latest estimate; -> command={command:.3f}")
        return {"actuator_command": command}


def main() -> None:
    environment = Environment(disturbance=Disturbance())
    plant = Plant("spacecraft")
    plant.add_state(State("attitude", dtype=float, initial=0.0))
    gyroscope = plant.add(Gyroscope())
    estimator = plant.add(Estimator())
    controller = plant.add(Controller())
    plant.auto_connect()
    plant.query(controller, environment["disturbance"], "torque")

    simulation = Simulation(environment=environment, plants=(plant,))

    print("\nSensor event triggers the estimator:")
    latest = estimator.step(gyroscope.step({}))
    controller.step(latest)

    print("\nEnvironmental model answers a query:")
    environment["disturbance"].query(
        "torque", {"attitude_error": 0.2}, context={"time_s": 0.0}
    )

    print("\nShowing Environment, Plant, state, blocks, and query/event links...")
    diagram = simulation.diagram(output="Z_BLOCKY/examples/1_block_definition")
    print(f"Diagram saved to {diagram}")


if __name__ == "__main__":
    main()
