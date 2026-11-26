use super::super::{derivative_vec, normalize_q, Runtime};

fn separate_add_scaled(y: &[f64], k: &[f64], scale: f64) -> Vec<f64> {
    y.iter().zip(k).map(|(a, b)| a + scale * b).collect()
}

fn attitude_derivative(y: &[f64], jd: f64, runtime: &Runtime) -> Result<Vec<f64>, String> {
    let mut derivative = derivative_vec(y, jd, runtime)?;
    // Separate propagation holds the orbit fixed during attitude RK stages.
    derivative[7..13].fill(0.0);
    Ok(derivative)
}

pub(crate) fn attitude_rk4(
    y: &[f64],
    jd: f64,
    dt: f64,
    runtime: &Runtime,
) -> Result<Vec<f64>, String> {
    let k1 = attitude_derivative(y, jd, runtime)?;
    let k2 = attitude_derivative(
        &separate_add_scaled(y, &k1, dt * 0.5),
        jd + dt * 0.5 / 86400.0,
        runtime,
    )?;
    let k3 = attitude_derivative(
        &separate_add_scaled(y, &k2, dt * 0.5),
        jd + dt * 0.5 / 86400.0,
        runtime,
    )?;
    let k4 = attitude_derivative(&separate_add_scaled(y, &k3, dt), jd + dt / 86400.0, runtime)?;
    let mut result: Vec<f64> = (0..y.len())
        .map(|i| y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
        .collect();
    normalize_q(&mut result)?;
    Ok(result)
}
