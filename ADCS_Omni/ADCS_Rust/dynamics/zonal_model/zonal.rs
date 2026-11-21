fn zonal_accel(p: [f64; 3], mu: f64, radius: f64, j: &[f64]) -> PyResult<[f64; 3]> {
    let [x, y, z] = p;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    if r == 0. {
        return Err(PyValueError::new_err(
            "Gravity is undefined at the body's centre",
        ));
    }
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
    Ok([
        base * x + mu_r3 * zg * (-s * x),
        base * y + mu_r3 * zg * (-s * y),
        base * z + mu_r3 * zg * (r - s * z),
    ])
}
#[pyfunction]
fn zonal_gravity_force(
    py: Python<'_>,
    position_i: &PyAny,
    mass: f64,
    epoch_jd: f64,
    gravitational_parameter: f64,
    reference_radius: f64,
    zonal_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let _ = epoch_jd;
    let p = vec3(py, position_i, "position_i")?;
    let j = vals(py, zonal_coefficients, "zonal_coefficients")?;
    Ok(out(
        py,
        scale(
            zonal_accel(p, gravitational_parameter, reference_radius, &j)?,
            mass,
        ),
    ))
}
