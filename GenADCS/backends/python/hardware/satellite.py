from dataclasses import dataclass, field

from .ADCS_subsystem import ADCSSubsystem
from .OBC_subsystem import OBCSubsystem
from .satellite_geometry import SatelliteGeometry
from .state_machine import StateMachine


@dataclass(slots=True)
class Satellite:
    geometry: SatelliteGeometry
    adcs: ADCSSubsystem = field(default_factory=ADCSSubsystem)
    obc: OBCSubsystem | None = None
    state_machine: StateMachine = field(default_factory=StateMachine)
