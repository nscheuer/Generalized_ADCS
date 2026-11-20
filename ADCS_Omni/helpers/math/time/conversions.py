"""Julian date and time-scale conversion kernels.

All scale conversion functions take and return Julian Date values in the named
scale. Calendar conversion uses the proleptic Gregorian calendar. The UTC/TAI
mapping is supported from 1972-01-01, when integral leap seconds began.
"""
import numpy as np
from numba import njit

_SECONDS_PER_DAY = 86400.0
_J2000 = 2451545.0

# Effective UTC Julian dates and corresponding TAI-UTC seconds. IERS leap
# second steps, through the latest step effective 2017-01-01.
_UTC_EFFECTIVE_JD = np.array([
    2441317.5, 2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5,
    2443144.5, 2443509.5, 2443874.5, 2444239.5, 2444786.5, 2445151.5,
    2445516.5, 2446247.5, 2447161.5, 2447892.5, 2448257.5, 2448804.5,
    2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5,
    2454832.5, 2456109.5, 2457204.5, 2457754.5,
], dtype=np.float64)
_TAI_MINUS_UTC = np.array([
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
], dtype=np.float64)


@njit(cache=True)
def calendar_to_jd(year, month, day, hour=0, minute=0, second=0.0):
    """Convert a proleptic Gregorian calendar date/time to Julian Date."""
    y = year
    m = month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd0 = np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + day + b - 1524.5
    return jd0 + (hour * 3600.0 + minute * 60.0 + second) / _SECONDS_PER_DAY


@njit(cache=True)
def jd_to_calendar(jd):
    """Convert a Julian Date to (year, month, day, hour, minute, second)."""
    zf = np.floor(jd + 0.5)
    f = jd + 0.5 - zf
    z = int(zf)
    alpha = int(np.floor((z - 1867216.25) / 36524.25))
    a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int(np.floor((b - 122.1) / 365.25))
    d = int(np.floor(365.25 * c))
    e = int(np.floor((b - d) / 30.6001))
    day_float = b - d - int(np.floor(30.6001 * e)) + f
    day = int(np.floor(day_float))
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    seconds = (day_float - day) * _SECONDS_PER_DAY
    hour = int(seconds // 3600.0)
    seconds -= hour * 3600.0
    minute = int(seconds // 60.0)
    second = seconds - minute * 60.0
    return year, month, day, hour, minute, second


@njit(cache=True)
def _tai_minus_utc(utc_jd):
    if utc_jd < _UTC_EFFECTIVE_JD[0]:
        raise ValueError("UTC/TAI conversion is supported from 1972-01-01")
    offset = _TAI_MINUS_UTC[0]
    for i in range(1, _UTC_EFFECTIVE_JD.size):
        if utc_jd < _UTC_EFFECTIVE_JD[i]:
            break
        offset = _TAI_MINUS_UTC[i]
    return offset


@njit(cache=True)
def utc_jd_to_tai_jd(utc_jd):
    """Convert UTC JD to TAI JD using the bundled leap-second table."""
    return utc_jd + _tai_minus_utc(utc_jd) / _SECONDS_PER_DAY


@njit(cache=True)
def tai_jd_to_utc_jd(tai_jd):
    """Convert TAI JD to UTC JD using fixed-point offset selection."""
    utc_jd = tai_jd - 10.0 / _SECONDS_PER_DAY
    for _ in range(3):
        utc_jd = tai_jd - _tai_minus_utc(utc_jd) / _SECONDS_PER_DAY
    return utc_jd


@njit(cache=True)
def tai_jd_to_tt_jd(tai_jd):
    """Convert TAI JD to TT JD (TT = TAI + 32.184 s)."""
    return tai_jd + 32.184 / _SECONDS_PER_DAY


@njit(cache=True)
def tt_jd_to_tai_jd(tt_jd):
    """Convert TT JD to TAI JD."""
    return tt_jd - 32.184 / _SECONDS_PER_DAY


@njit(cache=True)
def utc_jd_to_tt_jd(utc_jd):
    return tai_jd_to_tt_jd(utc_jd_to_tai_jd(utc_jd))


@njit(cache=True)
def tt_jd_to_utc_jd(tt_jd):
    return tai_jd_to_utc_jd(tt_jd_to_tai_jd(tt_jd))


@njit(cache=True)
def utc_jd_to_ut1_jd(utc_jd, dut1_seconds):
    """Convert UTC JD to UT1 JD; DUT1 is UT1 - UTC in SI seconds."""
    return utc_jd + dut1_seconds / _SECONDS_PER_DAY


@njit(cache=True)
def ut1_jd_to_utc_jd(ut1_jd, dut1_seconds):
    """Convert UT1 JD to UTC JD using caller-supplied DUT1 seconds."""
    return ut1_jd - dut1_seconds / _SECONDS_PER_DAY


@njit(cache=True)
def _tdb_minus_tt_seconds(tt_jd):
    # Dominant geocentric periodic terms; approximately millisecond accuracy.
    days = tt_jd - _J2000
    g = np.deg2rad(357.53 + 0.9856003 * days)
    return 0.001657 * np.sin(g) + 0.000022 * np.sin(2.0 * g)


@njit(cache=True)
def tdb_jd_to_tt_jd(tdb_jd):
    """Invert the compact TT/TDB approximation iteratively."""
    tt_jd = tdb_jd
    for _ in range(3):
        tt_jd = tdb_jd - _tdb_minus_tt_seconds(tt_jd) / _SECONDS_PER_DAY
    return tt_jd


@njit(cache=True)
def utc_jd_to_tdb_jd(utc_jd):
    return tt_jd_to_tdb_jd(utc_jd_to_tt_jd(utc_jd))


@njit(cache=True)
def tdb_jd_to_utc_jd(tdb_jd):
    return tt_jd_to_utc_jd(tdb_jd_to_tt_jd(tdb_jd))


@njit(cache=True)
def tai_jd_to_ut1_jd(tai_jd, dut1_seconds):
    return utc_jd_to_ut1_jd(tai_jd_to_utc_jd(tai_jd), dut1_seconds)


@njit(cache=True)
def ut1_jd_to_tai_jd(ut1_jd, dut1_seconds):
    return utc_jd_to_tai_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds))


@njit(cache=True)
def tt_jd_to_ut1_jd(tt_jd, dut1_seconds):
    return utc_jd_to_ut1_jd(tt_jd_to_utc_jd(tt_jd), dut1_seconds)


@njit(cache=True)
def ut1_jd_to_tt_jd(ut1_jd, dut1_seconds):
    return utc_jd_to_tt_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds))


@njit(cache=True)
def tt_jd_to_tdb_jd(tt_jd):
    """Convert TT JD to approximate TDB JD (compact periodic model)."""
    return tt_jd + _tdb_minus_tt_seconds(tt_jd) / _SECONDS_PER_DAY


@njit(cache=True)
def tai_jd_to_tdb_jd(tai_jd):
    return tt_jd_to_tdb_jd(tai_jd_to_tt_jd(tai_jd))


@njit(cache=True)
def tdb_jd_to_tai_jd(tdb_jd):
    return tt_jd_to_tai_jd(tdb_jd_to_tt_jd(tdb_jd))


@njit(cache=True)
def ut1_jd_to_tdb_jd(ut1_jd, dut1_seconds):
    return utc_jd_to_tdb_jd(ut1_jd_to_utc_jd(ut1_jd, dut1_seconds))


@njit(cache=True)
def tdb_jd_to_ut1_jd(tdb_jd, dut1_seconds):
    return utc_jd_to_ut1_jd(tdb_jd_to_utc_jd(tdb_jd), dut1_seconds)


@njit(cache=True)
def jd_to_utc_calendar_fields(jd):
    """Return Gregorian year, month, day, and seconds since UTC midnight."""
    year, month, day, hour, minute, second = jd_to_calendar(jd)
    return year, month, day, hour * 3600.0 + minute * 60.0 + second


@njit(cache=True)
def day_of_year(year, month, day):
    """Return the Gregorian day number within a year (January 1 is 1)."""
    lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    doy = day
    for i in range(month - 1):
        doy += lengths[i]
    if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        doy += 1
    return doy
