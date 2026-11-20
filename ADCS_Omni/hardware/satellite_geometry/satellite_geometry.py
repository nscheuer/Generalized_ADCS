from dataclasses import dataclass, field

from .satellite_panel import SatellitePanel


@dataclass(slots=True)
class SatelliteGeometry:
    mass: float
    inertia: object
    panels: list[SatellitePanel] = field(default_factory=list)
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
