use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{
    bias_py::{array_values, numpy_array},
    linear_drift::LinearDrift as NativeLinearDrift,
};

#[pyclass(name = "LinearDrift")]
pub struct LinearDrift {
    inner: NativeLinearDrift,
}

#[pymethods]
impl LinearDrift {
    #[new]
    #[pyo3(signature = (rate, initial_value=0.0))]
    fn new(rate: f64, initial_value: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeLinearDrift::new(rate, initial_value).map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }
    #[setter]
    fn set_rate(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeLinearDrift::new(value, self.inner.initial_value)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[getter]
    fn initial_value(&self) -> f64 {
        self.inner.initial_value
    }
    #[setter]
    fn set_initial_value(&mut self, value: f64) -> PyResult<()> {
        self.inner =
            NativeLinearDrift::new(self.inner.rate, value).map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(
        &self,
        py: Python<'_>,
        value: &PyAny,
        dt: f64,
        rng: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let _ = rng;
        let (values, shape) = array_values(py, value)?;
        let result = self
            .inner
            .apply(&values, dt)
            .map_err(PyValueError::new_err)?;
        numpy_array(py, &result, &shape)
    }
}
