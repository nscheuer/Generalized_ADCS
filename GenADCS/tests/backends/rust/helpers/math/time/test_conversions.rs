use super::*;
const SECONDS_PER_DAY: f64 = 86_400.0;
fn near(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b} (tol {tolerance})");
}

#[test]
fn known_calendar_dates_match_julian_dates() {
    let cases = [
        ((2000, 1, 1, 12, 0, 0.0), 2451545.0),
        ((1970, 1, 1, 0, 0, 0.0), 2440587.5),
        ((2000, 1, 1, 0, 0, 0.0), 2451544.5),
        ((2024, 2, 29, 0, 0, 0.0), 2460369.5),
        ((1900, 3, 1, 0, 0, 0.0), 2415079.5),
        ((2000, 3, 1, 0, 0, 0.0), 2451604.5),
    ];
    for ((y, m, d, h, mi, s), jd) in cases {
        near(calendar_to_jd(y, m, d, h, mi, s), jd, 2e-9);
    }
}

#[test]
fn calendar_roundtrips_and_julian_day_boundaries() {
    let cases = [
        (2000, 1, 1, 0, 0, 0.0),
        (2024, 2, 29, 23, 59, 59.25),
        (1969, 7, 20, 20, 17, 40.0),
        (1582, 10, 15, 0, 0, 0.0),
    ];
    for (y, m, d, h, mi, s) in cases {
        let (ry, rm, rd, rh, rmi, rs) = jd_to_calendar(calendar_to_jd(y, m, d, h, mi, s));
        assert_eq!((ry, rm, rd, rh, rmi), (y, m, d, h, mi));
        near(rs, s, 4e-5);
    }
    let midnight = calendar_to_jd(2025, 6, 1, 0, 0, 0.0);
    let (y, m, d, h, mi, _) = jd_to_calendar(midnight);
    assert_eq!((y, m, d, h, mi), (2025, 6, 1, 0, 0));
    let (y, m, d, h, mi, _) = jd_to_calendar(midnight + 0.5);
    assert_eq!((y, m, d, h, mi), (2025, 6, 1, 12, 0));
    let (y, m, d, h, mi, _) = jd_to_calendar(midnight + 1.0);
    assert_eq!((y, m, d, h, mi), (2025, 6, 2, 0, 0));
    near(
        calendar_to_jd(2020, 1, 1, 24, 0, 0.0),
        calendar_to_jd(2020, 1, 2, 0, 0, 0.0),
        1e-12,
    );
    near(
        calendar_to_jd(2020, 1, 1, 0, 60, 0.0),
        calendar_to_jd(2020, 1, 1, 1, 0, 0.0),
        1e-12,
    );
    near(
        calendar_to_jd(2020, 1, 1, 0, 0, 60.0),
        calendar_to_jd(2020, 1, 1, 0, 1, 0.0),
        1e-12,
    );
    let (y, m, d, seconds) = jd_to_utc_calendar_fields(calendar_to_jd(2024, 3, 4, 12, 34, 56.25));
    assert_eq!((y, m, d), (2024, 3, 4));
    near(seconds, 45_296.25, 4e-5);
}

#[test]
fn day_of_year_applies_gregorian_leap_rules() {
    for (y, m, d, n) in [
        (2024, 1, 1, 1),
        (2024, 2, 29, 60),
        (2023, 12, 31, 365),
        (2024, 12, 31, 366),
        (2000, 3, 1, 61),
        (1900, 3, 1, 60),
    ] {
        assert_eq!(day_of_year(y, m, d), n);
    }
}

#[test]
fn utc_tai_leap_second_effective_boundaries() {
    let effective = [
        2441317.5, 2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5, 2443144.5, 2443509.5,
        2443874.5, 2444239.5, 2444786.5, 2445151.5, 2445516.5, 2446247.5, 2447161.5, 2447892.5,
        2448257.5, 2448804.5, 2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5,
        2454832.5, 2456109.5, 2457204.5, 2457754.5,
    ];
    for (i, jd) in effective.iter().enumerate() {
        let after = utc_jd_to_tai_jd(*jd).unwrap();
        near((after - jd) * SECONDS_PER_DAY, 10.0 + i as f64, 2e-5);
        if i > 0 {
            let before = utc_jd_to_tai_jd(*jd - 1e-6).unwrap();
            near((after - before) * SECONDS_PER_DAY, 1.0864, 2e-4);
        }
    }
    let utc = calendar_to_jd(2024, 5, 6, 7, 8, 9.0);
    near(
        tai_jd_to_utc_jd(utc_jd_to_tai_jd(utc).unwrap()).unwrap(),
        utc,
        2e-10,
    );
    assert!(utc_jd_to_tai_jd(2441317.5 - 1e-5).is_err());
    assert!(utc_jd_to_tt_jd(2440000.0).is_err());
}

#[test]
fn tai_tt_ut1_tdb_conversions_roundtrip() {
    let tai = 2459000.123456;
    let tt = tai_jd_to_tt_jd(tai);
    near((tt - tai) * SECONDS_PER_DAY, 32.184, 2e-5);
    near(tt_jd_to_tai_jd(tt), tai, 2e-10);
    let utc = 2459000.25;
    for dut1 in [-0.8, 0.0, 0.45, 1.2] {
        let ut1 = utc_jd_to_ut1_jd(utc, dut1);
        near((ut1 - utc) * SECONDS_PER_DAY, dut1, 2e-5);
        near(ut1_jd_to_utc_jd(ut1, dut1), utc, 2e-10);
    }
    let dut1 = -0.137;
    let utc = calendar_to_jd(2022, 9, 17, 3, 4, 5.0);
    let tai = utc_jd_to_tai_jd(utc).unwrap();
    let tt = tai_jd_to_tt_jd(tai);
    let tdb = tt_jd_to_tdb_jd(tt);
    let ut1 = tai_jd_to_ut1_jd(tai, dut1).unwrap();
    near(tt_jd_to_tai_jd(tt), tai, 2e-10);
    near(tdb_jd_to_tt_jd(tdb), tt, 2e-10);
    near(tdb_jd_to_tai_jd(tdb), tai, 3e-10);
    near(tai_jd_to_tdb_jd(tai).unwrap(), tdb, 2e-10);
    near(ut1_jd_to_tai_jd(ut1, dut1).unwrap(), tai, 2e-10);
    near(ut1_jd_to_tt_jd(ut1, dut1).unwrap(), tt, 3e-10);
    near(tt_jd_to_ut1_jd(tt, dut1).unwrap(), ut1, 3e-10);
    near(ut1_jd_to_tdb_jd(ut1, dut1).unwrap(), tdb, 4e-10);
    near(tdb_jd_to_ut1_jd(tdb, dut1).unwrap(), ut1, 4e-10);
}

#[test]
fn tdb_periodic_correction_is_small_and_smooth() {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs: f64 = 0.0;
    let mut max_step: f64 = 0.0;
    let mut previous = 0.0;
    for i in 0..100 {
        let jd = 2451545.0 + 365.25 * i as f64 / 99.0;
        let c = (tt_jd_to_tdb_jd(jd) - jd) * SECONDS_PER_DAY;
        min = min.min(c);
        max = max.max(c);
        max_abs = max_abs.max(c.abs());
        if i > 0 {
            max_step = max_step.max((c - previous).abs());
        }
        previous = c;
    }
    assert!(max_abs < 0.002);
    assert!(max - min > 0.002);
    assert!(max_step < 0.001);
}
