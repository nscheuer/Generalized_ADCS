import pytest


@pytest.mark.parametrize(
    "date,jd",
    [
        ((2000, 1, 1, 12, 0, 0.0), 2451545.0),
        ((1970, 1, 1, 0, 0, 0.0), 2440587.5),
        ((2024, 2, 29, 0, 0, 0.0), 2460369.5),
        ((1900, 3, 1, 0, 0, 0.0), 2415079.5),
        ((2000, 3, 1, 0, 0, 0.0), 2451604.5),
    ],
)
def test_known_gregorian_dates(time_api, date, jd):
    assert time_api.calendar_to_jd(*date) == pytest.approx(jd, abs=2e-9)


@pytest.mark.parametrize(
    "date",
    [
        (2024, 2, 29, 23, 59, 59.25),
        (1969, 7, 20, 20, 17, 40.0),
        (1582, 10, 15, 0, 0, 0.0),
    ],
)
def test_julian_date_calendar_roundtrip(time_api, date):
    got = time_api.jd_to_calendar(time_api.calendar_to_jd(*date))
    assert got[:5] == date[:5]
    assert got[5] == pytest.approx(date[5], abs=4e-5)


def test_midday_midnight_rollover_and_seconds_of_day(time_api):
    midnight = time_api.calendar_to_jd(2025, 6, 1)
    assert time_api.jd_to_calendar(midnight)[:5] == (2025, 6, 1, 0, 0)
    assert time_api.jd_to_calendar(midnight + 0.5)[:5] == (2025, 6, 1, 12, 0)
    assert time_api.jd_to_calendar(midnight + 1.0)[:3] == (2025, 6, 2)
    assert time_api.calendar_to_jd(2020, 1, 1, 24) == pytest.approx(
        time_api.calendar_to_jd(2020, 1, 2), abs=2e-9
    )
    fields = time_api.jd_to_utc_calendar_fields(
        time_api.calendar_to_jd(2024, 3, 4, 12, 34, 56.25)
    )
    assert fields[:3] == (2024, 3, 4)
    assert fields[3] == pytest.approx(45296.25, abs=4e-5)


def test_gregorian_day_number_and_leap_centuries(time_api):
    for y, m, d, expected in [
        (2024, 2, 29, 60),
        (2023, 12, 31, 365),
        (2024, 12, 31, 366),
        (2000, 3, 1, 61),
        (1900, 3, 1, 60),
    ]:
        assert time_api.day_of_year(y, m, d) == expected
