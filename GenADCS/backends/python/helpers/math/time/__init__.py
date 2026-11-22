"""Numba-accelerated Julian date and astronomical time-scale conversions.

UTC conversions use the bundled TAI-UTC table (valid from 1972-01-01 through
its latest listed step, 2017-01-01). UT1 conversions require caller-supplied
DUT1 = UT1 - UTC in seconds. TDB uses a compact millisecond-level periodic
approximation; use a full ephemeris/SOFA implementation when higher accuracy
is needed.
"""
from .conversions import (
    calendar_to_jd,
    day_of_year,
    jd_to_calendar,
    jd_to_utc_calendar_fields,
    tai_jd_to_tdb_jd,
    tai_jd_to_tt_jd,
    tai_jd_to_utc_jd,
    tai_jd_to_ut1_jd,
    tdb_jd_to_utc_jd,
    tdb_jd_to_tai_jd,
    tdb_jd_to_tt_jd,
    tdb_jd_to_ut1_jd,
    tt_jd_to_tai_jd,
    tt_jd_to_ut1_jd,
    tt_jd_to_tdb_jd,
    tt_jd_to_utc_jd,
    ut1_jd_to_tdb_jd,
    utc_jd_to_tai_jd,
    utc_jd_to_tt_jd,
    utc_jd_to_ut1_jd,
    utc_jd_to_tdb_jd,
    ut1_jd_to_utc_jd,
    ut1_jd_to_tai_jd,
    ut1_jd_to_tt_jd,
)

__all__ = [name for name in globals() if not name.startswith("_")]
