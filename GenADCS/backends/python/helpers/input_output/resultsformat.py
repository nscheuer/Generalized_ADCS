"""Select recorded fields and full-dt sampling interval for a simulation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResultsFormat:
    attitude_quaternion: bool = True
    satellite_rate: bool = True
    wheel_momenta: bool = True
    N: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer")
        for name in ("attitude_quaternion", "satellite_rate", "wheel_momenta"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
