import pytest


def test_calendar_jd_and_decomposition_agree(time_backends):
    cases = [
        (2000, 1, 1, 12, 0, 0.0),
        (2024, 2, 29, 23, 59, 59.25),
        (1900, 3, 1, 0, 0, 0.0),
        (1969, 7, 20, 20, 17, 40.0),
    ]
    outputs = []
    for api in time_backends.values():
        outputs.append(
            [
                (
                    api.calendar_to_jd(*date),
                    api.jd_to_calendar(api.calendar_to_jd(*date)),
                    api.jd_to_utc_calendar_fields(api.calendar_to_jd(*date)),
                )
                for date in cases
            ]
        )
    for backend in outputs[1:]:
        for ref, actual in zip(outputs[0], backend):
            assert actual[0] == pytest.approx(ref[0], abs=2e-9)
            assert actual[1][:5] == ref[1][:5]
            assert actual[1][5] == pytest.approx(ref[1][5], abs=4e-5)
            assert actual[2][:3] == ref[2][:3]
            assert actual[2][3] == pytest.approx(ref[2][3], abs=4e-5)


def test_day_of_year_results_agree(time_backends):
    for y, m, d, expected in [
        (2024, 2, 29, 60),
        (1900, 3, 1, 60),
        (2000, 3, 1, 61),
        (2023, 12, 31, 365),
        (2024, 12, 31, 366),
    ]:
        vals = [api.day_of_year(y, m, d) for api in time_backends.values()]
        assert vals == [expected] * 3
