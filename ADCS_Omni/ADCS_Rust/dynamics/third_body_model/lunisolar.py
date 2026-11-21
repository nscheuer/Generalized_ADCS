"""Differential solar and lunar gravity force model."""
import numpy as np
from .._dynamics import lunisolar_acceleration
from ADCS_Omni.ADCS_Rust.environment.ephemeris import BODY_MOON, BODY_SUN
from ADCS_Omni.ADCS_Rust.helpers.math.time import utc_jd_to_tdb_jd

SUN_MU_M3_S2 = 1.32712440018e20
MOON_MU_M3_S2 = 4.9048695e12
KM_TO_M = 1000.0


class LunisolarForceModel:
    def __init__(self, geometry, ephemeris):
        if ephemeris is None or not hasattr(ephemeris, "position"):
            raise ValueError("lunisolar force requires an ephemeris with position()")
        self.mass = float(geometry.mass)
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("geometry.mass must be finite and positive")
        self.ephemeris = ephemeris

    def evaluate(self, position_i, epoch_utc_jd):
        position_i = np.ascontiguousarray(position_i, dtype=np.float64)
        if position_i.shape != (3,) or not np.all(np.isfinite(position_i)):
            raise ValueError("position_i must be a finite inertial 3-vector")
        epoch_tdb = utc_jd_to_tdb_jd(float(epoch_utc_jd))
        sun = KM_TO_M * self.ephemeris.position(BODY_SUN, epoch_tdb)
        moon = KM_TO_M * self.ephemeris.position(BODY_MOON, epoch_tdb)
        return self.mass * lunisolar_acceleration(position_i, sun, moon, SUN_MU_M3_S2, MOON_MU_M3_S2)

__all__ = ["LunisolarForceModel", "lunisolar_acceleration"]
