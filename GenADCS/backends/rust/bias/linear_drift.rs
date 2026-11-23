use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

#[pyclass]
pub struct LinearDrift { #[pyo3(get, set)] pub rate: f64, #[pyo3(get, set)] pub initial_value: f64 }

#[pymethods]
impl LinearDrift {
    #[new]
    #[pyo3(signature = (rate, initial_value=0.0))]
    fn new(rate: f64, initial_value: f64) -> Self { Self { rate, initial_value } }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, _value: &PyAny, _dt: f64, _rng: Option<&PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(""))
    }
}
