"""NRLMSISE-00 density from an inertial position and Julian date.

Positions are metres, Julian dates are UTC, and density is kg/m³. The model
uses default F10.7 = F10.7A = 150 sfu and Ap = 4 when indices are omitted.
The numerical model and coefficient tables are implemented in Numba modules.
"""

import numpy as np
from numba import njit

from ..atmospheric_model import AtmosphericModel
from ._nrlmsise00_core import msis_gtd7
from GenADCS.backends.python.helpers.math.frames import gcrf_position_to_geodetic
from GenADCS.backends.python.helpers.math.time import day_of_year, jd_to_utc_calendar_fields, utc_jd_to_tt_jd


@njit(cache=True)
def nrlmsise00_density_geodetic(day_of_year_utc, seconds_utc, altitude_km,
                                 latitude_deg, longitude_deg, f107=150.0,
                                 f107a=150.0, ap=4.0, effective_density=True):
    """Density [kg/m³] when geodetic location and UTC fields are already known."""
    local_solar_time = (seconds_utc / 3600.0 + longitude_deg / 15.0) % 24.0
    ap_history = np.full(7, -1.0)
    return msis_gtd7(day_of_year_utc, seconds_utc, altitude_km, latitude_deg,
                     longitude_deg, local_solar_time, f107a, f107, ap,
                     ap_history, effective_density).d[5]


@njit(cache=True)
def nrlmsise00_density(position_inertial_m, epoch_jd, f107=150.0,
                       f107a=150.0, ap=4.0, effective_density=True,
                       ap_array=None):
    """NRLMSISE-00 mass density [kg/m³] at an ECI position [m].

    ``f107`` is the preceding day's flux; ``f107a`` is the 81-day mean.
    The default includes anomalous oxygen in the drag density (``gtd7d``).
    Set ``effective_density=False`` for ordinary ``gtd7`` mass density.
    A seven-value ``ap_array`` enables the reference storm-time mode.
    """
    year, month, day, sec = jd_to_utc_calendar_fields(epoch_jd)
    doy = day_of_year(year, month, day)
    # JD input is UTC. DUT1 is unavailable here, so UT1≈UTC for this
    # low-order atmosphere location calculation.
    jd_tt = utc_jd_to_tt_jd(epoch_jd) if epoch_jd >= 2441317.5 else epoch_jd
    lat, lon, alt_km = gcrf_position_to_geodetic(
        position_inertial_m, epoch_jd, jd_tt
    )
    lst = (sec / 3600.0 + lon / 15.0) % 24.0
    if ap_array is None:
        ap_history = np.full(7, -1.0)
    else:
        if len(ap_array) != 7:
            raise ValueError("ap_array must contain seven Ap values")
        ap_history = ap_array
    return msis_gtd7(doy, sec, alt_km, lat, lon, lst, f107a, f107,
                     ap, ap_history, effective_density).d[5]


class NRLMSISE00(AtmosphericModel):
    """Full NRLMSISE-00 neutral-atmosphere density and co-rotation velocity."""

    density = staticmethod(nrlmsise00_density)
    density_geodetic = staticmethod(nrlmsise00_density_geodetic)


NRMLSISE = NRLMSISE00
