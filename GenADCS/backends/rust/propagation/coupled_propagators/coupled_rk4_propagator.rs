fn add_scaled(y: &[f64], k: &[f64], s: f64) -> [f64; 13] {
    std::array::from_fn(|i| y[i] + s * k[i])
}
pub fn rk4_vec(y: &[f64], jd: f64, dt: f64, p: &Runtime) -> Result<Vec<f64>, String> {
    let k1 = derivative_array(y, jd, p)?;
    let k2 = derivative_array(&add_scaled(y, &k1, dt * 0.5), jd + dt * 0.5 / 86400., p)?;
    let k3 = derivative_array(&add_scaled(y, &k2, dt * 0.5), jd + dt * 0.5 / 86400., p)?;
    let k4 = derivative_array(&add_scaled(y, &k3, dt), jd + dt / 86400., p)?;
    let mut out: Vec<f64> = (0..y.len())
        .map(|i| y[i] + dt * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]) / 6.)
        .collect();
    normalize_q(&mut out)?;
    Ok(out)
}
