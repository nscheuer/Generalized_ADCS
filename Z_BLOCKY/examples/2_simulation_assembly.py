"""Assemble a shared Environment and multiple independent Plants."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Z_BLOCKY import Block, Environment, Plant, Port, QueriedBlock, QueryPort, Simulation, State


class Gravity(QueriedBlock):
    def __init__(self) -> None:
        super().__init__("gravity", queries=(QueryPort("acceleration", tuple, tuple),))

    def query(self, port, request, context=None):
        x, y, z = request
        radius = (x * x + y * y + z * z) ** 0.5
        return tuple(-3.986004418e14 * value / radius**3 for value in (x, y, z))


class Orbit(Block):
    def __init__(self) -> None:
        super().__init__("orbit", frequency_hz=5.0, outputs=(Port("position_eci", tuple),))

    def step(self, inputs, context=None):
        return {"position_eci": inputs.get("position_eci", (7_000_000.0, 0.0, 0.0))}


class Navigation(Block):
    def __init__(self) -> None:
        super().__init__("navigation", frequency_hz=1.0, inputs=(Port("position_eci", tuple),))

    def step(self, inputs, context=None):
        print(f"Navigation received latest position: {inputs['position_eci']}")
        return {}


def build_plant(name: str, x_offset_m: float, gravity: QueriedBlock) -> Plant:
    plant = Plant(name)
    position = (7_000_000.0 + x_offset_m, 0.0, 0.0)
    plant.add_state(State("position_eci", tuple, position))
    plant.add_state(State("velocity_eci", tuple, (0.0, 7_500.0, 0.0)))
    orbit = plant.add(Orbit())
    navigation = plant.add(Navigation())
    plant.auto_connect()
    plant.query(orbit, gravity, "acceleration")
    return plant


def main() -> None:
    gravity = Gravity()
    environment = Environment(gravity=gravity, name="LEO")
    chief = build_plant("chief", 0.0, gravity)
    deputy = build_plant("deputy", 100.0, gravity)
    simulation = Simulation(environment=environment, plants=(chief, deputy), name="formation")

    print(f"Plants: {', '.join(plant.name for plant in simulation.plants)}")
    print(f"Event blocks: {len(simulation.blocks)}")
    print(f"Queried environmental models: {len(simulation.queried_blocks)}")
    print(f"Latest-value event routes: {simulation.connection_index}")
    print(f"On-demand query routes: {simulation.query_index}")
    print("No scheduler has run.")

    diagram = simulation.diagram(output="Z_BLOCKY/examples/2_simulation_assembly")
    print(f"Diagram saved to {diagram}")


if __name__ == "__main__":
    main()
