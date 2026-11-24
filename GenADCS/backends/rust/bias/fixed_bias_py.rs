use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{
    bias_py::{array_values, numpy_array},
    fixed_bias::FixedBias as NativeFixedBias,
};

#[pyclass(name = "FixedBias")]
pub struct FixedBias {
    inner: NativeFixedBias,
}

#[pymethods]
impl FixedBias {
    #[new]
    fn new(value: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeFixedBias::new(value).map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }
    #[setter]
    fn set_value(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeFixedBias::new(value).map_err(PyValueError::new_err)?;
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
        let _ = (dt, rng);
        let (values, shape) = array_values(py, value)?;
        numpy_array(py, &self.inner.apply(&values), &shape)
    }
}
