use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::dynamics_py::{numbers, py_array};
use super::{
    dop853_vec, integrate_dop853_native, integrate_rk4_native, rk4_vec, IntegrationError, Runtime,
};

fn integration_error(error: IntegrationError<PyErr>) -> PyErr {
    match error {
        IntegrationError::Native(message) => PyValueError::new_err(message),
        IntegrationError::Callback(error) => error,
    }
}

fn output(py: Python<'_>, times: Vec<f64>, states: Vec<Vec<f64>>) -> PyResult<PyObject> {
    let state_array = py.import("numpy")?.call_method1("asarray", (states,))?;
    Ok((times.to_pyarray(py), state_array).into_py(py))
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
