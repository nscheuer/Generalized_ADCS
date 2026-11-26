use numpy::{ndarray::Array2, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::dynamics_py::{numbers, py_array};
use super::sensor_sampling::{build_sampler_and_runtime, sampler_to_pydict};
use super::{
    dop853_vec, integrate_dop853_native, integrate_dop853_observed, integrate_rk4_native,
    integrate_rk4_observed, rk4_vec, IntegrationError, Runtime,
};

fn integration_error(error: IntegrationError<PyErr>) -> PyErr {
    match error {
        IntegrationError::Native(message) => PyValueError::new_err(message),
        IntegrationError::Callback(error) => error,
    }
}

fn output(py: Python<'_>, times: Vec<f64>, states: Vec<Vec<f64>>) -> PyResult<PyObject> {
    let width = states.first().map_or(0, Vec::len);
    let array = Array2::from_shape_vec(
        (states.len(), width),
        states.into_iter().flatten().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((times.into_pyarray(py), array.into_pyarray(py)).into_py(py))
}

#[pyfunction]
pub(super) fn rk4_step(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    runtime: &PyAny,
) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let result = rk4_vec(&y, jd, dt, &runtime).map_err(PyValueError::new_err)?;
    Ok(py_array(py, &result))
}

#[pyfunction]
pub(super) fn dop853_attempt(
    py: Python<'_>,
    y: &PyAny,
    f: &PyAny,
    jd: f64,
    h: f64,
    runtime: &PyAny,
    rtol: f64,
    atol: &PyAny,
) -> PyResult<(PyObject, PyObject, f64)> {
    let y = numbers(py, y)?;
    let f = numbers(py, f)?;
    let atol = numbers(py, atol)?;
    if f.len() != y.len() || atol.len() != y.len() {
        return Err(PyValueError::new_err("DOP853 vector lengths differ"));
    }
    let runtime = Runtime::from_py(py, runtime)?;
    let (next, next_derivative, error) =
        dop853_vec(&y, &f, jd, h, &runtime, rtol, &atol).map_err(PyValueError::new_err)?;
    Ok((py_array(py, &next), py_array(py, &next_derivative), error))
}

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,progress=false,progress_chunk_steps=500,record_every=1))]
pub(super) fn integrate_rk4(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let result = py.allow_threads(move || {
        let mut poll = || Python::with_gil(|py| py.check_signals());
        integrate_rk4_native(
            y,
            jd,
            dt,
            tf,
            &runtime,
            progress,
            progress_chunk_steps,
            record_every,
            &mut poll,
        )
    });
    let (times, states) = result.map_err(integration_error)?;
    output(py, times, states)
}

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,rtol,atol,max_step,min_step,progress=false,progress_chunk_steps=500,record_every=0))]
pub(super) fn integrate_dop853(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    rtol: f64,
    atol: &PyAny,
    max_step: f64,
    min_step: f64,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let atol = numbers(py, atol)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let result = py.allow_threads(move || {
        let mut poll = || Python::with_gil(|py| py.check_signals());
        integrate_dop853_native(
            y,
            jd,
            dt,
            tf,
            &runtime,
            rtol,
            atol,
            max_step,
            min_step,
            progress,
            progress_chunk_steps,
            record_every,
            &mut poll,
        )
    });
    let (times, states) = result.map_err(integration_error)?;
    output(py, times, states)
}

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,record_every,sensor_every,runtime,sensors,environment,seed=0,estimator=None,nees=false,adaptive=false,rtol=1e-9,atol=None,max_step=0.0,min_step=1e-6,progress=false,progress_chunk_steps=500))]
pub(super) fn simulate_coupled(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    record_every: usize,
    sensor_every: usize,
    runtime: &PyAny,
    sensors: &PyAny,
    environment: &PyAny,
    seed: u64,
    estimator: Option<&PyAny>,
    nees: bool,
    adaptive: bool,
    rtol: f64,
    atol: Option<&PyAny>,
    max_step: f64,
    min_step: f64,
    progress: bool,
    progress_chunk_steps: usize,
) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let atol_vec = match atol {
        Some(a) => numbers(py, a)?,
        None => vec![1e-12],
    };
    let max_step = if max_step <= 0.0 { dt } else { max_step };
    let (mut sampler, runtime) = build_sampler_and_runtime(
        py,
        jd,
        runtime,
        sensors,
        environment,
        seed,
        estimator,
        nees,
        false,
    )?;
    let sensor_interval = dt * sensor_every as f64;
    let result = py.allow_threads(move || {
        let mut poll = || Python::with_gil(|py| py.check_signals());
        let mut observe = |t: f64, state: &[f64], recorded: bool| -> Result<(), String> {
            let on_sensor = sensor_every > 0
                && (t / sensor_interval - (t / sensor_interval).round()).abs() < 1e-9;
            if on_sensor {
                let tick = (t / sensor_interval).round() as u64;
                sampler.sample(state, t, tick, &runtime)?;
            }
            if recorded {
                sampler.record(on_sensor);
            }
            Ok(())
        };
        let traj = if adaptive {
            integrate_dop853_observed(
                y,
                jd,
                dt,
                tf,
                &runtime,
                rtol,
                atol_vec,
                max_step,
                min_step,
                progress,
                progress_chunk_steps,
                record_every,
                &mut poll,
                &mut observe,
            )
        } else {
            integrate_rk4_observed(
                y,
                jd,
                dt,
                tf,
                &runtime,
                progress,
                progress_chunk_steps,
                record_every,
                &mut poll,
                &mut observe,
            )
        };
        traj.map(|(times, states)| (times, states, sampler))
    });
    let (times, states, sampler) = result.map_err(integration_error)?;
    let width = states.first().map_or(0, Vec::len);
    let array = Array2::from_shape_vec(
        (states.len(), width),
        states.into_iter().flatten().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let samples_dict = sampler_to_pydict(py, &sampler)?;
    Ok((times.into_pyarray(py), array.into_pyarray(py), samples_dict).into_py(py))
}
