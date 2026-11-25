from dataclasses import dataclass, field

import numpy as np

from .attitude_state import AttitudeState
from .internal_state import InternalState
from .orbital_state import OrbitalState


@dataclass(slots=True)
class State:
    attitude: AttitudeState
    orbit: OrbitalState
    internal: InternalState = field(default_factory=InternalState)

    def to_record(self, wheel_count=None):
        momentum = self.internal.wheel_momentum
        count = momentum.size if wheel_count is None else int(wheel_count)
        if count != momentum.size:
            raise ValueError(
                f"expected {count} reaction wheel momenta, got {momentum.size}"
            )
        dtype = np.dtype(
            [
                ("quaternion", np.float64, (4,)),
                ("rates", np.float64, (3,)),
                ("position", np.float64, (3,)),
                ("velocity", np.float64, (3,)),
                ("momentum", np.float64, (count,)),
            ]
        )
        record = np.empty(1, dtype=dtype)
        record["quaternion"][0] = self.attitude.quaternion_array
        record["rates"][0] = self.attitude.rates
        record["position"][0] = self.orbit.position
        record["velocity"][0] = self.orbit.velocity
        record["momentum"][0] = momentum
        return record

    @classmethod
    def from_record(cls, record, julian_date, ephemeris=None):
        value = record[0] if isinstance(record, np.ndarray) else record
        attitude = AttitudeState(value["quaternion"].copy(), value["rates"].copy())
        orbit = OrbitalState.from_cartesian(
            value["position"].copy(), value["velocity"].copy(), julian_date, ephemeris
        )
        internal = InternalState(value["momentum"].copy())
        return cls(attitude, orbit, internal)
