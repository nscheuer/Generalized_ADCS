from dataclasses import dataclass, field
from typing import cast

from .satellite_panel import SatellitePanel


@dataclass(slots=True)
class SatelliteGeometry:
    mass: float
    inertia: object
    panels: list[SatellitePanel] = field(
        default_factory=lambda: cast(list[SatellitePanel], [])
    )
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
