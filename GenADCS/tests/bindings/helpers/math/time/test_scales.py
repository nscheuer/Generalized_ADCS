import numpy as np
import pytest

SECONDS_PER_DAY = 86400.0


def test_all_scalar_scale_bindings_return_python_floats(time_api):
    utc = time_api.calendar_to_jd(2024, 5, 6, 7, 8, 9.0)
    tai = time_api.utc_jd_to_tai_jd(utc)
    tt = time_api.tai_jd_to_tt_jd(tai)
    tdb = time_api.tt_jd_to_tdb_jd(tt)
    ut1 = time_api.utc_jd_to_ut1_jd(utc, np.float64(-0.25))
    for value in (
        tai,
        tt,
        tdb,
        ut1,
        time_api.tt_jd_to_tai_jd(tt),
        time_api.tdb_jd_to_tt_jd(tdb),
        time_api.ut1_jd_to_utc_jd(ut1, -0.25),
    ):
        assert isinstance(value, float)


def test_binding_roundtrips_preserve_jd_precision(time_api):
    utc = time_api.calendar_to_jd(2023, 11, 14, 12, 13, 14.25)
    tai = time_api.utc_jd_to_tai_jd(utc)
    tt = time_api.utc_jd_to_tt_jd(utc)
    tdb = time_api.utc_jd_to_tdb_jd(utc)
    dut1 = 0.321
    ut1 = time_api.utc_jd_to_ut1_jd(utc, dut1)
    assert time_api.tai_jd_to_utc_jd(tai) == pytest.approx(utc, abs=2e-10)
    assert time_api.tt_jd_to_utc_jd(tt) == pytest.approx(utc, abs=2e-10)
    assert time_api.tdb_jd_to_utc_jd(tdb) == pytest.approx(utc, abs=3e-10)
    assert time_api.ut1_jd_to_utc_jd(ut1, dut1) == pytest.approx(utc, abs=2e-10)


def test_numpy_scalars_and_integer_inputs_are_accepted_for_scale_functions(time_api):
    utc = np.float64(2459000.25)
    dut1 = np.float64(-0.75)
    ut1 = time_api.utc_jd_to_ut1_jd(utc, dut1)
    assert ut1 == pytest.approx(float(utc) + float(dut1) / SECONDS_PER_DAY, abs=1e-12)
    assert time_api.ut1_jd_to_utc_jd(ut1, dut1) == pytest.approx(float(utc), abs=2e-10)
    assert time_api.tai_jd_to_tt_jd(np.int64(2459000)) == pytest.approx(
        2459000 + 32.184 / SECONDS_PER_DAY
    )


def test_unsupported_utc_epoch_maps_to_python_value_error(time_api):
    with pytest.raises(ValueError):
        time_api.utc_jd_to_tai_jd(2441317.5 - 1e-5)
    with pytest.raises(ValueError):
        time_api.utc_jd_to_tt_jd(2440000.0)


def test_bad_python_types_raise_type_error_at_binding_boundary(time_api):
    with pytest.raises((TypeError, ValueError)):
        time_api.tai_jd_to_tt_jd("2459000.0")
    with pytest.raises((TypeError, ValueError)):
        time_api.utc_jd_to_ut1_jd(2459000.0, "not dut1")
