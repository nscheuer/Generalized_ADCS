const SECS: f64 = 86400.0;
const LEAPS: [(f64, f64); 28] = [
    (2441317.5, 10.),
    (2441499.5, 11.),
    (2441683.5, 12.),
    (2442048.5, 13.),
    (2442413.5, 14.),
    (2442778.5, 15.),
    (2443144.5, 16.),
    (2443509.5, 17.),
    (2443874.5, 18.),
    (2444239.5, 19.),
    (2444786.5, 20.),
    (2445151.5, 21.),
    (2445516.5, 22.),
    (2446247.5, 23.),
    (2447161.5, 24.),
    (2447892.5, 25.),
    (2448257.5, 26.),
    (2448804.5, 27.),
    (2449169.5, 28.),
    (2449534.5, 29.),
    (2450083.5, 30.),
    (2450630.5, 31.),
    (2451179.5, 32.),
    (2453736.5, 33.),
    (2454832.5, 34.),
    (2456109.5, 35.),
    (2457204.5, 36.),
    (2457754.5, 37.),
];

pub fn tai_offset_native(jd: f64) -> Result<f64, &'static str> {
    if jd < LEAPS[0].0 {
        return Err("UTC/TAI conversion is supported from 1972-01-01");
    }
    Ok(LEAPS.iter().filter(|(e, _)| jd >= *e).last().unwrap().1)
}
pub fn tdb_tt_native(t: f64) -> f64 {
    let g = (357.53 + 0.9856003 * (t - 2451545.0)).to_radians();
    0.001657 * g.sin() + 0.000022 * (2. * g).sin()
}
pub fn calendar_to_jd_native(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    let (mut y, mut m) = (year, month);
    if m <= 2 {
        y -= 1;
        m += 12
    }
    let a = y.div_euclid(100);
    let b = 2 - a + a.div_euclid(4);
    ((365.25 * (y + 4716) as f64).floor()
        + (30.6001 * (m + 1) as f64).floor()
        + day as f64
        + b as f64
        - 1524.5)
        + (hour as f64 * 3600. + minute as f64 * 60. + second) / SECS
}
pub fn jd_to_calendar_native(jd: f64) -> (i32, i32, i32, i32, i32, f64) {
    let zf = (jd + 0.5).floor();
    let f = jd + 0.5 - zf;
    let z = zf as i32;
    let alpha = ((z as f64 - 1867216.25) / 36524.25).floor() as i32;
    let a = z + 1 + alpha - alpha.div_euclid(4);
    let b = a + 1524;
    let c = ((b as f64 - 122.1) / 365.25).floor() as i32;
    let d = (365.25 * c as f64).floor() as i32;
    let e = ((b - d) as f64 / 30.6001).floor() as i32;
    let dayf = (b - d - (30.6001 * e as f64).floor() as i32) as f64 + f;
    let day = dayf.floor() as i32;
    let month = if e < 14 { e - 1 } else { e - 13 };
    let year = if month > 2 { c - 4716 } else { c - 4715 };
    let mut s = (dayf - day as f64) * SECS;
    let h = (s / 3600.).floor() as i32;
    s -= h as f64 * 3600.;
    let m = (s / 60.).floor() as i32;
    s -= m as f64 * 60.;
    (year, month, day, h, m, s)
}
pub fn utc_jd_to_tai_jd_native(jd: f64) -> Result<f64, &'static str> {
    Ok(jd + tai_offset_native(jd)? / SECS)
}
pub fn tai_jd_to_utc_jd_native(jd: f64) -> Result<f64, &'static str> {
    let mut u = jd - 10. / SECS;
    for _ in 0..3 {
        u = jd - tai_offset_native(u)? / SECS
    }
    Ok(u)
}
pub fn tai_jd_to_tt_jd_native(jd: f64) -> f64 {
    jd + 32.184 / SECS
}
pub fn tt_jd_to_tai_jd_native(jd: f64) -> f64 {
    jd - 32.184 / SECS
}
pub fn utc_jd_to_tt_jd_native(jd: f64) -> Result<f64, &'static str> {
    Ok(tai_jd_to_tt_jd_native(utc_jd_to_tai_jd_native(jd)?))
}
pub fn tt_jd_to_utc_jd_native(jd: f64) -> Result<f64, &'static str> {
    tai_jd_to_utc_jd_native(tt_jd_to_tai_jd_native(jd))
}
pub fn utc_jd_to_ut1_jd_native(jd: f64, dut1_seconds: f64) -> f64 {
    jd + dut1_seconds / SECS
}
pub fn ut1_jd_to_utc_jd_native(jd: f64, dut1_seconds: f64) -> f64 {
    jd - dut1_seconds / SECS
}
pub fn tt_jd_to_tdb_jd_native(jd: f64) -> f64 {
    jd + tdb_tt_native(jd) / SECS
}
pub fn tdb_jd_to_tt_jd_native(jd: f64) -> f64 {
    let mut t = jd;
    for _ in 0..3 {
        t = jd - tdb_tt_native(t) / SECS
    }
    t
}
pub fn utc_jd_to_tdb_jd_native(jd: f64) -> Result<f64, &'static str> {
    Ok(tt_jd_to_tdb_jd_native(utc_jd_to_tt_jd_native(jd)?))
}
pub fn tdb_jd_to_utc_jd_native(jd: f64) -> Result<f64, &'static str> {
    tt_jd_to_utc_jd_native(tdb_jd_to_tt_jd_native(jd))
}
pub fn tai_jd_to_ut1_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    Ok(utc_jd_to_ut1_jd_native(tai_jd_to_utc_jd_native(jd)?, d))
}
pub fn ut1_jd_to_tai_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    utc_jd_to_tai_jd_native(ut1_jd_to_utc_jd_native(jd, d))
}
pub fn tt_jd_to_ut1_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    Ok(utc_jd_to_ut1_jd_native(tt_jd_to_utc_jd_native(jd)?, d))
}
pub fn ut1_jd_to_tt_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    utc_jd_to_tt_jd_native(ut1_jd_to_utc_jd_native(jd, d))
}
pub fn tai_jd_to_tdb_jd_native(jd: f64) -> Result<f64, &'static str> {
    Ok(tt_jd_to_tdb_jd_native(tai_jd_to_tt_jd_native(jd)))
}
pub fn tdb_jd_to_tai_jd_native(jd: f64) -> f64 {
    tt_jd_to_tai_jd_native(tdb_jd_to_tt_jd_native(jd))
}
pub fn ut1_jd_to_tdb_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    utc_jd_to_tdb_jd_native(ut1_jd_to_utc_jd_native(jd, d))
}
pub fn tdb_jd_to_ut1_jd_native(jd: f64, d: f64) -> Result<f64, &'static str> {
    Ok(utc_jd_to_ut1_jd_native(tdb_jd_to_utc_jd_native(jd)?, d))
}
pub fn jd_to_utc_calendar_fields_native(jd: f64) -> (i32, i32, i32, f64) {
    let (y, m, d, h, mi, s) = jd_to_calendar_native(jd);
    (y, m, d, h as f64 * 3600. + mi as f64 * 60. + s)
}
pub fn day_of_year_native(year: i32, month: i32, day: i32) -> i32 {
    let lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut n = day;
    for i in 0..(month - 1).max(0) as usize {
        if i < 12 {
            n += lengths[i]
        }
    }
    if month > 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        n += 1
    }
    n
}
