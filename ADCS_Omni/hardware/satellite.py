from dataclasses import dataclass, field

from .adcs_subsystem import ADCSSubsystem
from .satellite_geometry import SatelliteGeometry
from .state_machine import StateMachine


@dataclass(slots=True)
class Satellite:
    geometry: SatelliteGeometry
    adcs: ADCSSubsystem = field(default_factory=ADCSSubsystem)
    state_machine: StateMachine = field(default_factory=StateMachine)
