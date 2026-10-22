"""Bridge the viewer's UTC clock to Generalized ADCS planetary ephemerides."""

from datetime import datetime, timezone

import numpy as np

from ADCS.environment.ephemeris import DE442s


BODY_TARGETS = {
    'sun': 'sun',
    'mercury': 'mercury',
    'venus': 'venus',
    'moon': 'moon',
    'mars': 'mars barycenter',
    'jupiter': 'jupiter barycenter',
    'saturn': 'saturn barycenter',
    'uranus': 'uranus barycenter',
    'neptune': 'neptune barycenter',
}


class PlanetaryEphemerisTrack:
    """Sample Earth-relative ICRF positions in kilometres at a UTC epoch.

    DE442s gives outer-planet system barycenters rather than those planets'
    exact centers. Their positions are therefore approximate at close zoom.
    """

    def __init__(self, ephemeris=None) -> None:
        self._owns_ephemeris = ephemeris is None
        self.ephemeris = ephemeris if ephemeris is not None else DE442s()
        available = set(self.ephemeris.available_bodies)
        missing = [target for target in BODY_TARGETS.values() if target not in available]
        if missing:
            if self._owns_ephemeris:
                self.ephemeris.close()
            raise ValueError(f'Ephemeris lacks solar-system bodies: {missing}')

    def positions_at_utc(self, utc_seconds: float) -> dict[str, np.ndarray]:
        epoch = self.ephemeris.ts.from_datetime(
            datetime.fromtimestamp(utc_seconds, timezone.utc)
        )
        j2000 = (epoch.tt - 2451545.0) / 36525.0
        positions = {'earth': np.zeros(3, dtype=np.float64)}
        for name, target in BODY_TARGETS.items():
            positions[name] = np.asarray(
                self.ephemeris.position_eci(target, j2000,
                                            observer='earth', apparent=False),
                dtype=np.float64,
            )
        return positions

    def orbit_positions_at_utc(self, name: str, utc_seconds) -> np.ndarray:
        """Return a body's parent-relative positions at a vector of UTC times."""
        timestamps = np.asarray(utc_seconds, dtype=np.float64)
        datetimes = [datetime.fromtimestamp(float(value), timezone.utc)
                     for value in timestamps.reshape(-1)]
        epochs = self.ephemeris.ts.from_datetimes(datetimes)
        j2000 = (np.asarray(epochs.tt, dtype=np.float64) - 2451545.0) / 36525.0
        target = BODY_TARGETS.get(name, name)
        observer = 'earth' if name == 'moon' else 'sun'
        result = self.ephemeris.position_eci(
            target, j2000, observer=observer, apparent=False
        )
        return np.asarray(result, dtype=np.float64).reshape(timestamps.shape + (3,))

    def close(self) -> None:
        if self._owns_ephemeris:
            self.ephemeris.close()
