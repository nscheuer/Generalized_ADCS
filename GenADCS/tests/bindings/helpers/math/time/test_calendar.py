import gc
import numpy as np
import pytest


def test_calendar_keywords_defaults_and_python_scalar_types(time_api):
    assert time_api.calendar_to_jd(2000, 1, 1, hour=12) == pytest.approx(2451545.0)
    result = time_api.jd_to_calendar(2451545.0)
    assert isinstance(result, tuple) and len(result) == 6
    assert result[:5] == (2000, 1, 1, 12, 0)
    assert isinstance(result[-1], float)
    assert time_api.jd_to_utc_calendar_fields(2451545.0)[:3] == (2000, 1, 1)


def test_python_numeric_scalars_and_numpy_scalar_arguments_convert(time_api):
    jd = time_api.calendar_to_jd(np.int64(2024), np.int32(2), np.int64(29),
                                 np.int32(6), np.int64(7), np.float64(8.5))
    assert isinstance(jd, float)
    assert time_api.jd_to_calendar(jd)[:5] == (2024, 2, 29, 6, 7)
    assert time_api.jd_to_calendar(jd)[5] == pytest.approx(8.5, abs=4e-5)


@pytest.mark.parametrize("args", [(2000, 1), ("2000", 1, 1), (2000, 1, 1, "noon")])
def test_bad_calendar_binding_signatures_raise_python_errors(time_api, args):
    with pytest.raises((TypeError, ValueError)):
        time_api.calendar_to_jd(*args)


def test_binding_returns_owned_immutable_values_after_gc(time_api):
    returned = time_api.jd_to_calendar(time_api.calendar_to_jd(2024, 2, 29, 23, 59, 59.25))
    snapshot = tuple(returned)
    del returned
    gc.collect()
    assert snapshot[:5] == (2024, 2, 29, 23, 59)
    assert snapshot[5] == pytest.approx(59.25, abs=4e-5)


def test_day_of_year_binding_accepts_numpy_integer_scalars(time_api):
    assert time_api.day_of_year(np.int64(2024), np.int64(2), np.int64(29)) == 60
