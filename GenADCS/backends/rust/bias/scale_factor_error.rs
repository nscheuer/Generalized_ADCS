use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

#[pyclass]
pub struct ScaleFactorError { #[pyo3(get, set)] pub fraction: f64 }

#[pymethods]
impl ScaleFactorError {
    #[new]
    fn new(fraction: f64) -> Self { Self { fraction } }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, value: &PyAny, dt: f64, rng: Option<&PyAny>) -> PyResult<()> {
        let _ = (value, dt, rng);
        Err(PyNotImplementedError::new_err(""))
    }
}
