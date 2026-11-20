from dataclasses import dataclass, field

import numpy as np

from ADCS_Omni.environment.ephemeris import EphemerisModel


@dataclass(slots=True)
class OrbitalState:
    _position: object | None
    _velocity: object | None
    julian_date: float
    ephemeris: EphemerisModel | None = None
    _keplerian_elements: tuple | None = None
    _body_positions: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.julian_date = float(self.julian_date)
        if not np.isfinite(self.julian_date):
            raise ValueError("julian_date must be finite")
        if self._position is not None:
            self._position = self._vector(self._position, "position")
        if self._velocity is not None:
            self._velocity = self._vector(self._velocity, "velocity")

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return np.ascontiguousarray(vector)

    @classmethod
    def from_cartesian(cls, position, velocity, julian_date, ephemeris=None):
        return cls(position, velocity, julian_date, ephemeris)

    @classmethod
    def from_keplerian(cls, semi_major_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, true_anomaly, julian_date, ephemeris=None):
        elements = (semi_major_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, true_anomaly)
        return cls(None, None, julian_date, ephemeris, elements)

    def _load_cartesian(self):
        if self._position is None or self._velocity is None:
            raise NotImplementedError("Keplerian to Cartesian conversion is not implemented")

    @property
    def position(self):
        self._load_cartesian()
        return self._position

    @property
    def velocity(self):
        self._load_cartesian()
        return self._velocity

    def body_position(self, body):
        if body not in self._body_positions:
            if self.ephemeris is None:
                raise ValueError("an ephemeris must be configured")
            self._body_positions[body] = self.ephemeris.position(body, self.julian_date)
        return self._body_positions[body]

    @property
    def sun_position(self):
        return self.body_position("sun")

    @property
    def earth_position(self):
        return self.body_position("earth")

    @property
    def moon_position(self):
        return self.body_position("moon")
