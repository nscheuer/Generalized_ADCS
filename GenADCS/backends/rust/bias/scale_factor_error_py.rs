use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{
    bias_py::{array_values, numpy_array},
    scale_factor_error::ScaleFactorError as NativeScaleFactorError,
};

#[pyclass(name = "ScaleFactorError")]
pub struct ScaleFactorError {
    inner: NativeScaleFactorError,
}

#[pymethods]
impl ScaleFactorError {
    #[new]
    fn new(fraction: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeScaleFactorError::new(fraction).map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn fraction(&self) -> f64 {
        self.inner.fraction
    }
    #[setter]
    fn set_fraction(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeScaleFactorError::new(value).map_err(PyValueError::new_err)?;
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
