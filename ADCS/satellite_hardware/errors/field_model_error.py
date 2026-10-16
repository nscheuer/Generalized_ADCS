__all__ = ["FieldModelError"]

import numpy as np

from ADCS.orbits.universal_constants import TimeConstants


class FieldModelError:
    r"""
    A stand-in for the difference between the real geomagnetic field and the model.

    The estimator's field model (IGRF) and the simulation's "true" field are
    the same function, so without this a magnetometer is only ever wrong by
    its own noise. In reality the model misses the crustal and external
    fields, typically by 100 to 300 nT at low Earth orbit, and that error
    changes slowly as the spacecraft moves. This process adds such an error,
    in the inertial frame, to the field the plant's magnetometers see.

    Each inertial axis is an independent first-order Gauss-Markov process:
    it decays towards zero over ``correlation_time`` seconds and is topped
    up so that its standard deviation stays at ``std``. One instance is
    meant to be shared by all magnetometers of a spacecraft, since they are
    in the same field; it steps once per epoch however many sensors read it.

    Parameters
    ----------
    std : float
        Standard deviation of each axis of the error field, in tesla
        (the same unit the orbital state's field uses). Around ``2e-7`` is a
        reasonable value for a low Earth orbit.
    correlation_time : float
        Correlation time in seconds. The model error is a function of place
        more than of time, so a few hundred seconds, the time to move a few
        hundred kilometres, is a reasonable value.
    """

    def __init__(self, std: float, correlation_time: float) -> None:
        self.std = float(std)
        self.correlation_time = float(correlation_time)
        if not np.isfinite(self.std) or self.std < 0.0:
            raise ValueError("std must be finite and non-negative")
        if not np.isfinite(self.correlation_time) or self.correlation_time <= 0.0:
            raise ValueError("correlation_time must be positive and finite")
        self.eci = np.zeros(3)
        self.last_update_time = float("nan")

    def __bool__(self) -> bool:
        return self.std > 0.0

    def copy(self) -> "FieldModelError":
        result = FieldModelError(self.std, self.correlation_time)
        result.eci = self.eci.copy()
        result.last_update_time = self.last_update_time
        return result

    def _update(self, j2000: float) -> None:
        """Advance the error field to ``j2000`` (Julian centuries); a repeated epoch is a no-op."""
        if not self.std:
            return
        if not np.isfinite(self.last_update_time):
            # Start in the stationary distribution rather than at zero.
            self.eci = np.random.normal(0.0, self.std, size=3)
            self.last_update_time = j2000
            return
        elapsed = (j2000 - self.last_update_time) * TimeConstants.cent2sec
        if elapsed <= 0.0:
            return
        decay = np.exp(-elapsed / self.correlation_time)
        self.eci = decay * self.eci + np.random.normal(0.0, self.std * np.sqrt(1.0 - decay * decay), size=3)
        self.last_update_time = j2000
