from dataclasses import dataclass

from ADCS_Omni.helpers.math import DCM, Quaternion


@dataclass(slots=True)
class AttitudeState:
    orientation: Quaternion | DCM
    rates: object

    def __post_init__(self):
        if isinstance(self.orientation, DCM):
            self.orientation = self.orientation.to_quaternion()
        elif not isinstance(self.orientation, Quaternion):
            self.orientation = Quaternion(self.orientation)
