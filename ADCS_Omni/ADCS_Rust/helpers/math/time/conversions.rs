fn tai_offset(jd: f64) -> PyResult<f64> {
    if jd < LEAPS[0].0 {
        return Err(PyValueError::new_err(
            "UTC/TAI conversion is supported from 1972-01-01",
        ));
    }
    Ok(LEAPS.iter().filter(|(e, _)| jd >= *e).last().unwrap().1)
}
fn tdb_tt(t: f64) -> f64 {
    let g = (357.53 + 0.9856003 * (t - 2451545.0)).to_radians();
    0.001657 * g.sin() + 0.000022 * (2. * g).sin()
}
#[pyfunction]
#[pyo3(signature=(year,month,day,hour=0,minute=0,second=0.0))]
fn calendar_to_jd(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
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
#[pyfunction]
fn jd_to_calendar(jd: f64) -> (i32, i32, i32, i32, i32, f64) {
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
#[pyfunction]
fn utc_jd_to_tai_jd(jd: f64) -> PyResult<f64> {
    Ok(jd + tai_offset(jd)? / SECS)
}
#[pyfunction]
fn tai_jd_to_utc_jd(jd: f64) -> PyResult<f64> {
    let mut u = jd - 10. / SECS;
    for _ in 0..3 {
        u = jd - tai_offset(u)? / SECS
    }
    Ok(u)
}
#[pyfunction]
fn tai_jd_to_tt_jd(jd: f64) -> f64 {
    jd + 32.184 / SECS
}
#[pyfunction]
fn tt_jd_to_tai_jd(jd: f64) -> f64 {
    jd - 32.184 / SECS
}
#[pyfunction]
fn utc_jd_to_tt_jd(jd: f64) -> PyResult<f64> {
    Ok(tai_jd_to_tt_jd(utc_jd_to_tai_jd(jd)?))
}
#[pyfunction]
fn tt_jd_to_utc_jd(jd: f64) -> PyResult<f64> {
    tai_jd_to_utc_jd(tt_jd_to_tai_jd(jd))
}
#[pyfunction]
fn utc_jd_to_ut1_jd(jd: f64, dut1_seconds: f64) -> f64 {
    jd + dut1_seconds / SECS
}
#[pyfunction]
fn ut1_jd_to_utc_jd(jd: f64, dut1_seconds: f64) -> f64 {
    jd - dut1_seconds / SECS
}
#[pyfunction]
fn tt_jd_to_tdb_jd(jd: f64) -> f64 {
    jd + tdb_tt(jd) / SECS
}
#[pyfunction]
fn tdb_jd_to_tt_jd(jd: f64) -> f64 {
    let mut t = jd;
    for _ in 0..3 {
        t = jd - tdb_tt(t) / SECS
    }
    t
}
#[pyfunction]
fn utc_jd_to_tdb_jd(jd: f64) -> PyResult<f64> {
    Ok(tt_jd_to_tdb_jd(utc_jd_to_tt_jd(jd)?))
}
#[pyfunction]
fn tdb_jd_to_utc_jd(jd: f64) -> PyResult<f64> {
    tt_jd_to_utc_jd(tdb_jd_to_tt_jd(jd))
}
#[pyfunction]
fn tai_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64> {
    Ok(utc_jd_to_ut1_jd(tai_jd_to_utc_jd(jd)?, d))
}
#[pyfunction]
fn ut1_jd_to_tai_jd(jd: f64, d: f64) -> PyResult<f64> {
    utc_jd_to_tai_jd(ut1_jd_to_utc_jd(jd, d))
}
#[pyfunction]
fn tt_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64> {
    Ok(utc_jd_to_ut1_jd(tt_jd_to_utc_jd(jd)?, d))
}
#[pyfunction]
fn ut1_jd_to_tt_jd(jd: f64, d: f64) -> PyResult<f64> {
    utc_jd_to_tt_jd(ut1_jd_to_utc_jd(jd, d))
}
#[pyfunction]
fn tai_jd_to_tdb_jd(jd: f64) -> PyResult<f64> {
    Ok(tt_jd_to_tdb_jd(tai_jd_to_tt_jd(jd)))
}
#[pyfunction]
fn tdb_jd_to_tai_jd(jd: f64) -> f64 {
    tt_jd_to_tai_jd(tdb_jd_to_tt_jd(jd))
}
#[pyfunction]
fn ut1_jd_to_tdb_jd(jd: f64, d: f64) -> PyResult<f64> {
    utc_jd_to_tdb_jd(ut1_jd_to_utc_jd(jd, d))
}
#[pyfunction]
fn tdb_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64> {
    Ok(utc_jd_to_ut1_jd(tdb_jd_to_utc_jd(jd)?, d))
}
#[pyfunction]
fn jd_to_utc_calendar_fields(jd: f64) -> (i32, i32, i32, f64) {
    let (y, m, d, h, mi, s) = jd_to_calendar(jd);
    (y, m, d, h as f64 * 3600. + mi as f64 * 60. + s)
}
#[pyfunction]
fn day_of_year(year: i32, month: i32, day: i32) -> i32 {
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
