use numpy::ToPyArray;
use pyo3::prelude::*;

use super::super::{dynamics_py, Runtime};
use super::reference_attitude_propagator::set_reference;

#[pyfunction]
#[pyo3(signature=(y,jd,mode,runtime))]
pub(crate) fn set_reference_attitude(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    mode: i32,
    runtime: &PyAny,
) -> PyResult<PyObject> {
    let mut y = dynamics_py::numbers(py, y)?;
    let ephemeris_path: Option<String> = if mode == 2 {
        Some(runtime.getattr("ephemeris_path")?.extract()?)
    } else {
        None
    };
    let mut runtime = Runtime::from_py(py, runtime)?;
    if let Some(path) = ephemeris_path {
        if runtime.ephemeris.is_none() {
            runtime.ephemeris = Some(
                ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                    .map_err(pyo3::exceptions::PyValueError::new_err)?,
            );
        }
    }
    set_reference(&mut y, jd, mode, &runtime).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(y.to_pyarray(py).into_py(py))
}
