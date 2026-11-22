fn add_scaled(y: &[f64], k: &[f64], s: f64) -> Vec<f64> {
    y.iter().zip(k).map(|(a, b)| a + s * b).collect()
}
pub fn rk4_vec(y: &[f64], jd: f64, dt: f64, p: &Runtime) -> PyResult<Vec<f64>> {
    let k1 = derivative_vec(y, jd, p)?;
    let k2 = derivative_vec(&add_scaled(y, &k1, dt * 0.5), jd + dt * 0.5 / 86400., p)?;
    let k3 = derivative_vec(&add_scaled(y, &k2, dt * 0.5), jd + dt * 0.5 / 86400., p)?;
    let k4 = derivative_vec(&add_scaled(y, &k3, dt), jd + dt / 86400., p)?;
    let mut out: Vec<f64> = (0..y.len())
        .map(|i| y[i] + dt * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]) / 6.)
        .collect();
    normalize_q(&mut out)?;
    Ok(out)
}
#[pyfunction]
fn rk4_step(py: Python<'_>, y: &PyAny, jd: f64, dt: f64, runtime: &PyAny) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    Ok(py_array(py, &rk4_vec(&y, jd, dt, &runtime)?))
}
