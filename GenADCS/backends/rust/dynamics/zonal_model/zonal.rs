fn zonal_acceleration(position_i: [f64; 3], mu: f64, radius: f64, j: &[f64]) -> [f64; 3] {
    let [x, y, z] = position_i;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let s = z / r;
    let ratio = radius / r;
    let mut rn = ratio * ratio;
    let ps = [
        0.5 * (3. * s * s - 1.),
        0.5 * (5. * s.powi(3) - 3. * s),
        (35. * s.powi(4) - 30. * s * s + 3.) / 8.,
        (63. * s.powi(5) - 70. * s.powi(3) + 15. * s) / 8.,
        (231. * s.powi(6) - 315. * s.powi(4) + 105. * s * s - 5.) / 16.,
    ];
    let dp = [
        3. * s,
        0.5 * (15. * s * s - 3.),
        (140. * s.powi(3) - 60. * s) / 8.,
        (315. * s.powi(4) - 210. * s * s + 15.) / 8.,
        (1386. * s.powi(5) - 1260. * s.powi(3) + 210. * s) / 16.,
    ];
    let mut radial = 1.;
    let mut zg = 0.;
    for k in 0..j.len().min(5) {
        radial -= (k as f64 + 3.) * j[k] * rn * ps[k];
        zg += j[k] * rn * dp[k];
        rn *= ratio;
    }
    let mu_r3 = mu / (r2 * r);
    let base = -mu_r3 * radial;
    // The latitude term enters with a negative sign:
    // -mu/r^3 * sum(Jn (R/r)^n Pn'(s)) * (r z_hat - s r_vec).
    [
        base * x - mu_r3 * zg * (-s * x),
        base * y - mu_r3 * zg * (-s * y),
        base * z - mu_r3 * zg * (r - s * z),
    ]
}

/// Compute the force on a body due to a central gravity field with zonal
/// harmonic coefficients J2 through J6.
pub fn zonal_gravity_force_native(
    position_i: [f64; 3],
    mass: f64,
    _epoch_jd: f64,
    gravitational_parameter: f64,
    reference_radius: f64,
    zonal_coefficients: &[f64],
) -> Result<[f64; 3], &'static str> {
    if position_i
        .iter()
        .chain([mass, gravitational_parameter, reference_radius, _epoch_jd].iter())
        .chain(zonal_coefficients.iter())
        .any(|value| !value.is_finite())
    {
        return Err("zonal gravity inputs must be finite");
    }
    if position_i == [0.; 3] {
        return Err("Gravity is undefined at the body's centre");
    }
    let force = scale(
        zonal_acceleration(
            position_i,
            gravitational_parameter,
            reference_radius,
            zonal_coefficients,
        ),
        mass,
    );
    if force.iter().any(|value| !value.is_finite()) {
        return Err("zonal gravity force must be finite");
    }
    Ok(force)
}
