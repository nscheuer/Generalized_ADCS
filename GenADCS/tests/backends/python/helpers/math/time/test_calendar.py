import pytest

from GenADCS.backends.python.helpers.math.time import (
    calendar_to_jd,
    day_of_year,
    jd_to_calendar,
    jd_to_utc_calendar_fields,
)


@pytest.mark.parametrize(
    "date, expected",
    [
        ((2000, 1, 1, 12, 0, 0.0), 2451545.0),
        ((1970, 1, 1, 0, 0, 0.0), 2440587.5),
        ((2000, 1, 1, 0, 0, 0.0), 2451544.5),
        ((2024, 2, 29, 0, 0, 0.0), 2460369.5),
        ((1900, 3, 1, 0, 0, 0.0), 2415079.5),
        ((2000, 3, 1, 0, 0, 0.0), 2451604.5),
    ],
)
def test_calendar_to_known_julian_dates(date, expected):
    assert calendar_to_jd(*date) == pytest.approx(expected, abs=2e-9)


@pytest.mark.parametrize(
    "date",
    [
        (2000, 1, 1, 0, 0, 0.0),
        (2024, 2, 29, 23, 59, 59.25),
        (1969, 7, 20, 20, 17, 40.0),
        (1582, 10, 15, 0, 0, 0.0),
    ],
)
def test_calendar_jd_roundtrip_calendar_fields(date):
    result = jd_to_calendar(calendar_to_jd(*date))
    assert result[:5] == date[:5]
    assert result[5] == pytest.approx(date[5], abs=4e-5)


def test_midnight_noon_and_day_boundaries():
    midnight = calendar_to_jd(2025, 6, 1)
    assert midnight % 1 == pytest.approx(0.5)
    assert jd_to_calendar(midnight)[:6] == (2025, 6, 1, 0, 0, 0.0)
    assert jd_to_calendar(midnight + 0.5)[:5] == (2025, 6, 1, 12, 0)
    assert jd_to_calendar(midnight + 1)[:3] == (2025, 6, 2)


def test_subday_arguments_carry_naturally():
    assert calendar_to_jd(2020, 1, 1, 24) == pytest.approx(calendar_to_jd(2020, 1, 2))
    assert calendar_to_jd(2020, 1, 1, 0, 60) == pytest.approx(
        calendar_to_jd(2020, 1, 1, 1)
    )
    assert calendar_to_jd(2020, 1, 1, 0, 0, 60.0) == pytest.approx(
        calendar_to_jd(2020, 1, 1, 0, 1)
    )


def test_utc_calendar_fields_are_seconds_from_midnight():
    jd = calendar_to_jd(2024, 3, 4, 12, 34, 56.25)
    year, month, day, seconds = jd_to_utc_calendar_fields(jd)
    assert (year, month, day) == (2024, 3, 4)
    assert seconds == pytest.approx(12 * 3600 + 34 * 60 + 56.25, abs=4e-5)


@pytest.mark.parametrize(
    "year, month, day, expected",
    [
        (2024, 1, 1, 1),
        (2024, 2, 29, 60),
        (2023, 12, 31, 365),
        (2024, 12, 31, 366),
        (2000, 3, 1, 61),
        (1900, 3, 1, 60),
    ],
)
def test_day_of_year_gregorian_leap_rules(year, month, day, expected):
    assert day_of_year(year, month, day) == expected
