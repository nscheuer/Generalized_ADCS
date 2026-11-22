from dataclasses import dataclass


@dataclass(slots=True)
class PlanetaryBody:
    name: str
    gravitational_parameter: float
    reference_radius: float
