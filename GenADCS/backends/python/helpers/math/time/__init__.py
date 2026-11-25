"""Numba-accelerated Julian date and astronomical time-scale conversions.

UTC conversions use the bundled TAI-UTC table (valid from 1972-01-01 through
its latest listed step, 2017-01-01). UT1 conversions require caller-supplied
DUT1 = UT1 - UTC in seconds. TDB uses a compact millisecond-level periodic
approximation; use a full ephemeris/SOFA implementation when higher accuracy
is needed.
"""

from .conversions import (
    calendar_to_jd as calendar_to_jd,
)
from .conversions import (
    day_of_year as day_of_year,
)
from .conversions import (
    jd_to_calendar as jd_to_calendar,
)
from .conversions import (
    jd_to_utc_calendar_fields as jd_to_utc_calendar_fields,
)
from .conversions import (
    tai_jd_to_tdb_jd as tai_jd_to_tdb_jd,
)
from .conversions import (
    tai_jd_to_tt_jd as tai_jd_to_tt_jd,
)
from .conversions import (
    tai_jd_to_ut1_jd as tai_jd_to_ut1_jd,
)
from .conversions import (
    tai_jd_to_utc_jd as tai_jd_to_utc_jd,
)
from .conversions import (
    tdb_jd_to_tai_jd as tdb_jd_to_tai_jd,
)
from .conversions import (
    tdb_jd_to_tt_jd as tdb_jd_to_tt_jd,
)
from .conversions import (
    tdb_jd_to_ut1_jd as tdb_jd_to_ut1_jd,
)
from .conversions import (
    tdb_jd_to_utc_jd as tdb_jd_to_utc_jd,
)
from .conversions import (
    tt_jd_to_tai_jd as tt_jd_to_tai_jd,
)
from .conversions import (
    tt_jd_to_tdb_jd as tt_jd_to_tdb_jd,
)
from .conversions import (
    tt_jd_to_ut1_jd as tt_jd_to_ut1_jd,
)
from .conversions import (
    tt_jd_to_utc_jd as tt_jd_to_utc_jd,
)
from .conversions import (
    ut1_jd_to_tai_jd as ut1_jd_to_tai_jd,
)
from .conversions import (
    ut1_jd_to_tdb_jd as ut1_jd_to_tdb_jd,
)
from .conversions import (
    ut1_jd_to_tt_jd as ut1_jd_to_tt_jd,
)
from .conversions import (
    ut1_jd_to_utc_jd as ut1_jd_to_utc_jd,
)
from .conversions import (
    utc_jd_to_tai_jd as utc_jd_to_tai_jd,
)
from .conversions import (
    utc_jd_to_tdb_jd as utc_jd_to_tdb_jd,
)
from .conversions import (
    utc_jd_to_tt_jd as utc_jd_to_tt_jd,
)
from .conversions import (
    utc_jd_to_ut1_jd as utc_jd_to_ut1_jd,
)

__all__ = [name for name in globals() if not name.startswith("_")]
