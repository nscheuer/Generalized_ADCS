from dataclasses import dataclass


@dataclass(slots=True)
class SatellitePanel:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    dimensions: tuple[float, float]
    eta_s: float = 0.0
    eta_d: float = 0.0
    eta_a: float = 1.0
    cd: float = 2.2
