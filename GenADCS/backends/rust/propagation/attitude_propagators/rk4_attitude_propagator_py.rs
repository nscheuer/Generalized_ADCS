use numpy::ToPyArray;
use pyo3::prelude::*;

use super::super::{dynamics_py, Runtime};
use super::rk4_attitude_propagator::attitude_rk4;

#[pyfunction]
#[pyo3(signature=(y,jd,dt,runtime))]
pub(crate) fn rk4_attitude_step(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    dt: f64,
    runtime: &PyAny,
) -> PyResult<PyObject> {
    let y = dynamics_py::numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let result =
        attitude_rk4(&y, jd, dt, &runtime).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(result.to_pyarray(py).into_py(py))
}
