import pytest

SEC = 86400.0


def test_time_scale_conversion_graph_matches(time_backends):
    utc_by_backend = []
    for api in time_backends.values():
        utc = api.calendar_to_jd(2022, 9, 17, 3, 4, 5.0)
        dut1 = -0.137
        tai = api.utc_jd_to_tai_jd(utc)
        tt = api.utc_jd_to_tt_jd(utc)
        tdb = api.utc_jd_to_tdb_jd(utc)
        ut1 = api.utc_jd_to_ut1_jd(utc, dut1)
        utc_by_backend.append(
            (
                utc,
                tai,
                tt,
                tdb,
                ut1,
                api.ut1_jd_to_tdb_jd(ut1, dut1),
                api.tdb_jd_to_ut1_jd(tdb, dut1),
            )
        )
    ref = utc_by_backend[0]
    for vals in utc_by_backend[1:]:
        for actual, expected in zip(vals, ref):
            assert actual == pytest.approx(expected, abs=4e-10)


def test_leap_table_offsets_match_at_every_transition(time_backends):
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
    for i, jd in enumerate(effective):
        offsets = [
            (api.utc_jd_to_tai_jd(jd) - jd) * SEC for api in time_backends.values()
        ]
        for offset in offsets:
            assert offset == pytest.approx(10.0 + i, abs=2e-5)
