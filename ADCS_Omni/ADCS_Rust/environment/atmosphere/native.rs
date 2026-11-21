pub fn native_density(position: [f64; 3], jd: f64) -> PyResult<f64> {
    native_density_with(position, jd, 150., 150., 4., [-1.; 7], true)
}
pub fn native_density_with(
    position: [f64; 3],
    jd: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    history: [f64; 7],
    effective: bool,
) -> PyResult<f64> {
    let (year, month, day, seconds) = native_calendar(jd);
    let doy = day_of_year(year, month, day);
    let rot = magnetic_native::native_rotation(jd)?;
    let fixed = [
        rot[0][0] * position[0] + rot[0][1] * position[1] + rot[0][2] * position[2],
        rot[1][0] * position[0] + rot[1][1] * position[1] + rot[1][2] * position[2],
        rot[2][0] * position[0] + rot[2][1] * position[1] + rot[2][2] * position[2],
    ];
    let geo = native_geodetic(fixed);
    Ok(evaluate_geodetic(
        doy, seconds, geo[2], geo[0], geo[1], f107, f107a, ap, history, effective,
    ))
}
pub fn native_corotation(position: [f64; 3]) -> [f64; 3] {
    [-7.2921150e-5 * position[1], 7.2921150e-5 * position[0], 0.]
}
fn native_calendar(jd: f64) -> (i32, i32, i32, f64) {
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
    (year, month, day, (dayf - day as f64) * 86400.)
}
fn day_of_year(year: i32, month: i32, day: i32) -> i32 {
    let days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut sum = day;
    for m in 0..(month - 1).max(0) as usize {
        sum += days[m];
    }
    if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
        sum += 1;
    }
    sum
}
fn native_geodetic(p: [f64; 3]) -> [f64; 3] {
    let a = 6378137.;
    let e2: f64 = 6.6943799901413165e-3;
    let rho = p[0].hypot(p[1]);
    let lon = p[1].atan2(p[0]);
    let (lat, alt) = if rho == 0. {
        (
            if p[2] >= 0. {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            },
            p[2].abs() - a * (1. - e2).sqrt(),
        )
    } else {
        let mut lat = p[2].atan2(rho * (1. - e2));
        for _ in 0..8 {
            let s = lat.sin();
            let n = a / (1. - e2 * s * s).sqrt();
            lat = (p[2] + e2 * n * s).atan2(rho);
        }
        let s = lat.sin();
        let n = a / (1. - e2 * s * s).sqrt();
        (lat, rho / lat.cos() - n)
    };
    [lat.to_degrees(), lon.to_degrees(), alt / 1000.]
}
