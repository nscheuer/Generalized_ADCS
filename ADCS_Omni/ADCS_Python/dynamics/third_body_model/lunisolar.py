"""Differential solar and lunar gravity using an Earth-centered ephemeris."""

import numpy as np
from numba import njit

from ADCS_Omni.ADCS_Python.environment.ephemeris import BODY_MOON, BODY_SUN, ephemeris_position
from ADCS_Omni.ADCS_Python.helpers.math.time.conversions import utc_jd_to_tdb_jd

SUN_MU_M3_S2 = 1.32712440018e20
MOON_MU_M3_S2 = 4.9048695e12
KM_TO_M = 1000.0


@njit(cache=True)
def lunisolar_acceleration(position_i, sun_position_i, moon_position_i, sun_mu, moon_mu):
    """Return third-body acceleration [m/s²] in inertial axes.

    Positions are Earth-centered inertial metres. The indirect term accounts
    for acceleration of the Earth-centered origin by each third body.
    """
    acceleration = np.zeros(3, dtype=np.float64)
    for body_position, mu in ((sun_position_i, sun_mu), (moon_position_i, moon_mu)):
        relative = body_position - position_i
        body_distance = np.sqrt(np.dot(body_position, body_position))
        relative_distance = np.sqrt(np.dot(relative, relative))
        if body_distance == 0.0 or relative_distance == 0.0:
            raise ValueError("third-body distance must be nonzero")
        acceleration += mu * (
            relative / relative_distance**3 - body_position / body_distance**3
        )
    return acceleration


@njit(cache=True)
def _lunisolar_force_from_ephemeris(position_i, epoch_utc_jd, segments, times, data, mass):
    epoch_tdb = utc_jd_to_tdb_jd(epoch_utc_jd)
    sun_i = KM_TO_M * ephemeris_position(BODY_SUN, epoch_tdb, segments, times, data)
    moon_i = KM_TO_M * ephemeris_position(BODY_MOON, epoch_tdb, segments, times, data)
    return mass * lunisolar_acceleration(
        position_i, sun_i, moon_i, SUN_MU_M3_S2, MOON_MU_M3_S2,
    )


class LunisolarForceModel:
    """Solar and lunar third-body force using prepared ephemeris arrays.

    The ephemeris and spacecraft mass are captured at construction, so the
    hot path only passes numerical values into the compiled evaluator.
    """

    def __init__(self, geometry, ephemeris):
        if ephemeris is None or not all(hasattr(ephemeris, name) for name in ("_segments", "_times", "_data")):
            raise ValueError("lunisolar force requires an ephemeris with prepared SPK arrays")
        self.mass = float(geometry.mass)
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("geometry.mass must be finite and positive")
        self.segments = ephemeris._segments
        self.times = ephemeris._times
        self.data = ephemeris._data

    def evaluate(self, position_i, epoch_utc_jd):
        position_i = np.ascontiguousarray(position_i, dtype=np.float64)
        if position_i.shape != (3,) or not np.all(np.isfinite(position_i)):
            raise ValueError("position_i must be a finite inertial 3-vector")
        return _lunisolar_force_from_ephemeris(
            position_i, float(epoch_utc_jd), self.segments, self.times,
            self.data, self.mass,
        )
