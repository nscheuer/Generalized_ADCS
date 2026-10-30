"""Precomputed SGP4 orbit propagation from TLE elements."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numba import njit
from sgp4.api import Satrec, WGS72
from skyfield.api import EarthSatellite
from tqdm import tqdm

from ADCS.environment.environment import Environment
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


# SGP4's conventional effective reference-density factor, expressed in the
# inverse-Earth-radius units used by B*. It is 2.461e-5 * 6378.136, with the
# radius value in kilometres as used by the SGP4 coefficient convention.
_SGP4_REFERENCE_DENSITY_KG_M2_PER_EARTH_RADIUS = 0.15696615


@njit(cache=True)
def _isotropic_mean_cd_area_kernel(areas: np.ndarray, coefficients: np.ndarray) -> float:
    """Compute the uniform-attitude mean ``CD * projected area``."""
    weighted_surface_area = 0.0
    for index in range(areas.size):
        weighted_surface_area += areas[index] * coefficients[index]
    return weighted_surface_area * 0.25


class SGP4Orbit(Orbit):
    """Precompute an SGP4 trajectory from TLE data.

    Pass either a Skyfield ``EarthSatellite`` or a pair of TLE lines. TLE mean
    elements are propagated using the WGS-72 SGP4 model, and Skyfield converts
    the resulting TEME samples to the package's inertial frame. When a
    ``satellite`` is supplied, its ``Drag_Disturbance`` geometry is reduced to
    an isotropic mean projected area and used to estimate a physical B* value.
    This one-time average is efficient, but does not represent a commanded,
    time-varying attitude profile. ``bstar`` explicitly overrides both the
    geometry estimate and the TLE value. Without either ``bstar`` or a
    ``satellite``, the TLE's fitted B* is preserved. SGP4 uses its own empirical
    drag model and B*; the supplied environment atmosphere populates the
    returned ``Orbital_State`` samples but does not drive SGP4 propagation.

    ``end_time`` and optional ``start_time`` are Julian centuries since J2000;
    ``dt`` is in seconds, matching :class:`Orbit`.

    References
    ----------
    * SGP4 theory and its reference implementation: Vallado et al.,
      `AIAA-2006-6753 <https://celestrak.org/publications/AIAA/2006-6753/>`_.
    * TLE fields and the meaning and units of B*: `CelesTrak TLE FAQ
      <https://celestrak.org/columns/v04n03/>`_.
    * The B* reference-density conversion uses the convention documented in
      Koehler et al., `NASA/TM-20250010561
      <https://ntrs.nasa.gov/citations/20250010561>`_.
    """

    def __init__(
        self,
        tle,
        end_time: float,
        dt: float,
        *,
        line2: str | None = None,
        start_time: float | None = None,
        satellite=None,
        bstar: float | None = None,
        environment: Environment | None = None,
        verbose: bool = True,
    ) -> None:
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(end_time):
            raise ValueError("end_time must be finite")

        environment = environment or Environment()
        if isinstance(tle, EarthSatellite):
            if line2 is not None:
                raise TypeError("line2 cannot be supplied with an EarthSatellite")
            source = tle
        elif isinstance(tle, str):
            if line2 is None:
                raise TypeError("line2 is required when tle is a TLE line 1 string")
            source = EarthSatellite(tle, line2, ts=environment.ephemeris.ts)
        elif isinstance(tle, Sequence) and len(tle) == 2:
            if line2 is not None:
                raise TypeError("line2 cannot be supplied when tle is a pair of lines")
            source = EarthSatellite(tle[0], tle[1], ts=environment.ephemeris.ts)
        else:
            raise TypeError("tle must be an EarthSatellite, a TLE line 1, or a pair of TLE lines")

        model = source.model
        mean_cd_area = None
        if satellite is not None:
            mass = float(satellite.mass)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("satellite.mass must be finite and positive")
            mean_cd_area = self._isotropic_mean_cd_area(satellite)
        if bstar is not None:
            effective_bstar = float(bstar)
        elif satellite is not None:
            effective_bstar = (
                0.5 * mean_cd_area / mass
                * _SGP4_REFERENCE_DENSITY_KG_M2_PER_EARTH_RADIUS
            )
        else:
            mean_cd_area = None
            effective_bstar = float(model.bstar)
        if not np.isfinite(effective_bstar):
            raise ValueError("bstar must be finite")

        # SGP4 computes drag-dependent coefficients at initialization, so make
        # a new record instead of mutating the read-only Satrec.bstar field.
        propagated_model = Satrec()
        epoch = model.jdsatepoch + model.jdsatepochF - 2433281.5
        propagated_model.sgp4init(
            WGS72,
            model.operationmode,
            model.satnum,
            epoch,
            effective_bstar,
            model.ndot,
            model.nddot,
            model.ecco,
            model.argpo,
            model.inclo,
            model.mo,
            model.no_kozai,
            model.nodeo,
        )
        self.satellite = EarthSatellite.from_satrec(
            propagated_model, environment.ephemeris.ts
        )
        self.bstar = effective_bstar
        self.mean_cd_area_m2 = mean_cd_area if satellite is not None else None

        if start_time is None:
            start_time = float((self.satellite.epoch.tt - 2451545.0) / 36525.0)
        start_time = float(start_time)
        end_time = float(end_time)
        if end_time < start_time:
            raise ValueError("end_time must be at or after start_time")

        duration_s = (end_time - start_time) * TimeConstants.cent2sec
        count = int(np.floor(duration_s / dt))
        times = start_time + np.arange(count + 1) * dt * TimeConstants.sec2cent
        if not np.isclose(times[-1], end_time, rtol=0.0, atol=1e-12):
            times = np.append(times, end_time)

        skyfield_times = environment.ephemeris.ts.tt_jd(
            times * 36525.0 + 2451545.0
        )
        propagated = self.satellite.at(skyfield_times)
        position = np.asarray(propagated.position.km, dtype=float)
        velocity = np.asarray(propagated.velocity.km_per_s, dtype=float)
        if position.ndim == 1:
            position = position.reshape(3, 1)
            velocity = velocity.reshape(3, 1)

        states = []
        for index, time in enumerate(tqdm(
            times,
            desc="Propagating SGP4 Orbit",
            unit="step",
            disable=not verbose,
        )):
            states.append(Orbital_State.from_position(
                J2000=float(time),
                R=position[:, index],
                V=velocity[:, index],
                environment=environment,
            ))
        super().__init__(states, environment=environment, verbose=False)

    @staticmethod
    def _isotropic_mean_cd_area(satellite) -> float:
        """Return ``sum(CD * area) / 4`` across standard drag geometries."""
        from ADCS.satellite_hardware.disturbances import Drag_Disturbance

        weighted_surface_area = 0.0
        for disturbance in satellite.disturbances:
            if isinstance(disturbance, Drag_Disturbance) and disturbance.active:
                weighted_surface_area += _isotropic_mean_cd_area_kernel(
                    np.asarray(disturbance.areas, dtype=np.float64),
                    np.asarray(disturbance.CDs, dtype=np.float64),
                )
        if weighted_surface_area <= 0:
            raise ValueError(
                "satellite must have at least one Drag_Disturbance with positive area and CD"
            )
        return weighted_surface_area

