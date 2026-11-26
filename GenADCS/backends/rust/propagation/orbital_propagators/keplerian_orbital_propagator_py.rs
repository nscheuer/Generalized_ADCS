use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::keplerian_orbital_propagator::keplerian_step_native;

#[pyfunction]
pub(crate) fn keplerian_step(
    position: Vec<f64>,
    velocity: Vec<f64>,
    mu: f64,
    dt: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if position.len() != 3 || velocity.len() != 3 {
        return Err(PyValueError::new_err(
            "position and velocity must have three components",
        ));
    }
    if !mu.is_finite() || mu <= 0.0 || !dt.is_finite() {
        return Err(PyValueError::new_err("mu must be positive and dt finite"));
    }
    let (p, v) = keplerian_step_native(
        position.try_into().unwrap(),
        velocity.try_into().unwrap(),
        mu,
        dt,
    )
    .map_err(PyValueError::new_err)?;
    Ok((p.to_vec(), v.to_vec()))
}

#[pyfunction]
pub(crate) fn integrate_keplerian(
    y: Vec<f64>,
    mu: f64,
    dt: f64,
    tf: f64,
    record_every: usize,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if y.len() < 13
        || !mu.is_finite()
        || mu <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || !tf.is_finite()
        || tf < 0.0
        || record_every == 0
    {
        return Err(PyValueError::new_err(
            "invalid Keplerian integration arguments",
        ));
    }
    let full = (tf / dt).floor() as usize;
    let mut rem = tf - full as f64 * dt;
    if rem <= 1e-12 * 1.0_f64.max(tf) {
        rem = 0.0;
    }
    let total = full + usize::from(rem > 0.0);
    let mut times = vec![0.0];
    let mut states = vec![y.clone()];
    let mut state = y;
    for step in 0..total {
        let step_dt = if step == full { rem } else { dt };
        let (p, v) = keplerian_step_native(
            state[7..10].try_into().unwrap(),
            state[10..13].try_into().unwrap(),
            mu,
            step_dt,
        )
        .map_err(PyValueError::new_err)?;
        state[7..10].copy_from_slice(&p);
        state[10..13].copy_from_slice(&v);
        let completed = step + 1;
        if completed % record_every == 0 || completed == total {
            times.push((completed as f64 * dt).min(tf));
            states.push(state.clone());
        }
    }
    Ok((times, states))
}

pub(crate) fn register_keplerian(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keplerian_step, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_keplerian, m)?)?;
    Ok(())
}
