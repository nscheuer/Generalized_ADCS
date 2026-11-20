from dataclasses import dataclass


@dataclass(slots=True)
class AttitudeState:
    orientation: object
    rates: object
