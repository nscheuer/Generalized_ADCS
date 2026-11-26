use numpy::{ndarray::Array2, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::dynamics_py::numbers;
use super::separate_solver::integrate_separate_native;
use super::Runtime;

#[pyfunction]
#[pyo3(signature=(y,jd,dt,tf,runtime,reference_mode=-1,progress=false,progress_chunk_steps=500,record_every=1))]
pub(super) fn integrate_separate(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    tf: f64,
    runtime: &PyAny,
    reference_mode: i32,
    progress: bool,
    progress_chunk_steps: usize,
    record_every: usize,
) -> PyResult<PyObject> {
    if progress_chunk_steps == 0 {
        return Err(PyValueError::new_err(
            "progress_chunk_steps must be positive",
        ));
    }
    let y = numbers(py, y)?;
    let ephemeris_path: Option<String> = if reference_mode == 2 {
        Some(runtime.getattr("ephemeris_path")?.extract()?)
    } else {
        None
    };
    let mut runtime = Runtime::from_py(py, runtime)?;
    if let Some(path) = ephemeris_path {
        if runtime.ephemeris.is_none() {
            runtime.ephemeris = Some(
                ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                    .map_err(PyValueError::new_err)?,
            );
        }
    }
    let (result, signal_error) = py.allow_threads(move || {
        let mut signal_error = None;
        let result = integrate_separate_native(
            y,
            jd,
            dt,
            tf,
            &runtime,
            reference_mode,
            progress,
            progress_chunk_steps,
            record_every,
            || {
                Python::with_gil(|py| match py.check_signals() {
                    Ok(()) => true,
                    Err(error) => {
                        signal_error = Some(error);
                        false
                    }
                })
            },
        );
        (result, signal_error)
    });
    if let Some(error) = signal_error {
        return Err(error);
    }
    let (times, states) = result.map_err(PyValueError::new_err)?;
    let width = states.first().map_or(0, Vec::len);
    let array = Array2::from_shape_vec(
        (states.len(), width),
        states.into_iter().flatten().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((times.into_pyarray(py), array.into_pyarray(py)).into_py(py))
}
