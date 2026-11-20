from dataclasses import dataclass


@dataclass(slots=True)
class Propagator:
    attitude: object | None = None
    orbital: object | None = None
    coupled: object | None = None

    def __post_init__(self):
        if (self.coupled is None and (self.attitude is None or self.orbital is None)) or (self.coupled is not None and (self.attitude is not None or self.orbital is not None)):
            raise ValueError("provide a coupled propagator or both attitude and orbital propagators")
