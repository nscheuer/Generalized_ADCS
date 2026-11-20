from dataclasses import dataclass, field

from .attitude_state import AttitudeState
from .internal_state import InternalState
from .orbital_state import OrbitalState


@dataclass(slots=True)
class State:
    attitude: AttitudeState
    orbit: OrbitalState
    internal: InternalState = field(default_factory=InternalState)
