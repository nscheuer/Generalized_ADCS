from dataclasses import dataclass, field
from typing import cast

import numpy as np

from GenADCS.backends.python.environment.ephemeris import (
    BODY_EARTH,
    BODY_MOON,
    BODY_SUN,
    EphemerisModel,
)


@dataclass(slots=True)
class OrbitalState:
    _position: np.ndarray | None
    _velocity: np.ndarray | None
    julian_date: float
    ephemeris: EphemerisModel | None = None
    _keplerian_elements: tuple[float, float, float, float, float, float] | None = None
    _body_positions: dict[str | int, np.ndarray] = field(
        default_factory=lambda: cast(dict[str | int, np.ndarray], {}),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.julian_date = float(self.julian_date)
        if not np.isfinite(self.julian_date):
            raise ValueError("julian_date must be finite")
        if self._position is not None:
            self._position = self._vector(self._position, "position")
        if self._velocity is not None:
            self._velocity = self._vector(self._velocity, "velocity")

    @staticmethod
    def _vector(value: object, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return np.ascontiguousarray(vector)

    @classmethod
    def from_cartesian(
        cls,
        position: object,
        velocity: object,
        julian_date: float,
        ephemeris: EphemerisModel | None = None,
    ) -> "OrbitalState":
        return cls(cast(np.ndarray, position), cast(np.ndarray, velocity), julian_date, ephemeris)

    @classmethod
    def from_keplerian(
        cls,
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        ascending_node: float,
        argument_of_periapsis: float,
        true_anomaly: float,
        julian_date: float,
        ephemeris: EphemerisModel | None = None,
    ) -> "OrbitalState":
        elements = (
            semi_major_axis,
            eccentricity,
            inclination,
            ascending_node,
            argument_of_periapsis,
            true_anomaly,
        )
        return cls(None, None, julian_date, ephemeris, elements)

    def _load_cartesian(self) -> None:
        if self._position is None or self._velocity is None:
            raise NotImplementedError(
                "Keplerian to Cartesian conversion is not implemented"
            )

    @property
    def position(self) -> np.ndarray:
        self._load_cartesian()
        assert self._position is not None
        return self._position

    @property
    def velocity(self) -> np.ndarray:
        self._load_cartesian()
        assert self._velocity is not None
        return self._velocity

    def body_position(self, body: str | int) -> np.ndarray:
        if body not in self._body_positions:
            if self.ephemeris is None:
                raise ValueError("an ephemeris must be configured")
            body_ids = {"sun": BODY_SUN, "earth": BODY_EARTH, "moon": BODY_MOON}
            body_id = body_ids[body.lower()] if isinstance(body, str) else int(body)
            self._body_positions[body] = self.ephemeris.position(
                body_id, self.julian_date
            )
        return self._body_positions[body]

    @property
    def sun_position(self) -> np.ndarray:
        return self.body_position("sun")

    @property
    def earth_position(self) -> np.ndarray:
        return self.body_position("earth")

    @property
    def moon_position(self) -> np.ndarray:
        return self.body_position("moon")
