import numpy as np
import pytest

from GenADCS.backends.python.helpers.math import time as ts

SECONDS_PER_DAY = 86400.0


def test_fixed_tai_tt_offset_is_32_184_seconds() -> None:
    tai = 2459000.123456
    tt = ts.tai_jd_to_tt_jd(tai)
    assert (tt - tai) * SECONDS_PER_DAY == pytest.approx(32.184, abs=2e-5)
    assert ts.tt_jd_to_tai_jd(tt) == pytest.approx(tai, abs=2e-10)


def test_utc_tai_leap_table_boundaries_and_roundtrip() -> None:
    # Each effective date advances TAI-UTC by one second.
    effective = [
        2441317.5,
        2441499.5,
        2441683.5,
        2442048.5,
        2442413.5,
        2442778.5,
        2443144.5,
        2443509.5,
        2443874.5,
        2444239.5,
        2444786.5,
        2445151.5,
        2445516.5,
        2446247.5,
        2447161.5,
        2447892.5,
        2448257.5,
        2448804.5,
        2449169.5,
        2449534.5,
        2450083.5,
        2450630.5,
        2451179.5,
        2453736.5,
        2454832.5,
        2456109.5,
        2457204.5,
        2457754.5,
    ]
    for index, jd in enumerate(effective):
        after = ts.utc_jd_to_tai_jd(jd)
        assert (after - jd) * SECONDS_PER_DAY == pytest.approx(10.0 + index, abs=2e-5)
        if index:
            before = ts.utc_jd_to_tai_jd(jd - 1e-6)
            assert (after - before) * SECONDS_PER_DAY == pytest.approx(1.0864, abs=2e-4)
    utc = ts.calendar_to_jd(2024, 5, 6, 7, 8, 9.0)
    assert ts.tai_jd_to_utc_jd(ts.utc_jd_to_tai_jd(utc)) == pytest.approx(
        utc, abs=2e-10
    )


def test_utc_before_supported_epoch_is_rejected() -> None:
    with pytest.raises(ValueError):
        ts.utc_jd_to_tai_jd(2441317.5 - 1e-5)
    with pytest.raises(ValueError):
        ts.utc_jd_to_tt_jd(2440000.0)


def test_ut1_utc_dut1_sign_and_inverse() -> None:
    utc = 2459000.25
    for dut1 in (-0.8, 0.0, 0.45, 1.2):
        ut1 = ts.utc_jd_to_ut1_jd(utc, dut1)
        assert (ut1 - utc) * SECONDS_PER_DAY == pytest.approx(dut1, abs=2e-5)
        assert ts.ut1_jd_to_utc_jd(ut1, dut1) == pytest.approx(utc, abs=2e-10)


def test_all_named_scale_paths_are_mutual_inverses() -> None:
    dut1 = -0.137
    tai = ts.utc_jd_to_tai_jd(ts.calendar_to_jd(2022, 9, 17, 3, 4, 5.0))
    tt = ts.tai_jd_to_tt_jd(tai)
    tdb = ts.tt_jd_to_tdb_jd(tt)
    ut1 = ts.tai_jd_to_ut1_jd(tai, dut1)
    assert ts.tt_jd_to_tai_jd(tt) == pytest.approx(tai, abs=2e-10)
    assert ts.tdb_jd_to_tt_jd(tdb) == pytest.approx(tt, abs=2e-10)
    assert ts.tdb_jd_to_tai_jd(tdb) == pytest.approx(tai, abs=3e-10)
    assert ts.tai_jd_to_tdb_jd(tai) == pytest.approx(tdb, abs=2e-10)
    assert ts.ut1_jd_to_tai_jd(ut1, dut1) == pytest.approx(tai, abs=2e-10)
    assert ts.ut1_jd_to_tt_jd(ut1, dut1) == pytest.approx(tt, abs=3e-10)
    assert ts.tt_jd_to_ut1_jd(tt, dut1) == pytest.approx(ut1, abs=3e-10)
    assert ts.ut1_jd_to_tdb_jd(ut1, dut1) == pytest.approx(tdb, abs=4e-10)
    assert ts.tdb_jd_to_ut1_jd(tdb, dut1) == pytest.approx(ut1, abs=4e-10)


def test_tdb_periodic_correction_is_millisecond_scale_and_smooth() -> None:
    values = np.linspace(2451545.0, 2451545.0 + 365.25, 100)
    corrections = np.array(
        [(ts.tt_jd_to_tdb_jd(float(x)) - x) * SECONDS_PER_DAY for x in values]
    )
    assert np.max(np.abs(corrections)) < 0.002
    assert np.ptp(corrections) > 0.002
    assert np.max(np.abs(np.diff(corrections))) < 0.001
