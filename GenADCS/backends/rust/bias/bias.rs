use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct Bias { inner: PyObject }

#[pymethods]
impl Bias {
    #[new]
    #[pyo3(signature = (bias=None, std_bias=None, bounds=None))]
    fn new(py: Python<'_>, bias: Option<PyObject>, std_bias: Option<PyObject>, bounds: Option<PyObject>) -> PyResult<Self> {
        let cls = py.import("GenADCS.backends.python.bias.bias")?.getattr("Bias")?;
        let kwargs = PyDict::new(py);
        if let Some(bias) = bias { kwargs.set_item("bias", bias)?; }
        if let Some(std_bias) = std_bias { kwargs.set_item("std_bias", std_bias)?; }
        if let Some(bounds) = bounds { kwargs.set_item("bounds", bounds)?; }
        Ok(Self { inner: cls.call((), Some(kwargs))?.into() })
    }

    #[getter(bias)]
    fn bias_value(&self, py: Python<'_>) -> PyResult<PyObject> { Ok(self.inner.as_ref(py).getattr("bias")?.into()) }
    #[setter]
    fn set_bias(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> { self.inner.as_ref(py).setattr("bias", value) }
    #[getter]
    fn std_bias(&self, py: Python<'_>) -> PyResult<PyObject> { Ok(self.inner.as_ref(py).getattr("std_bias")?.into()) }
    #[setter]
    fn set_std_bias(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> { self.inner.as_ref(py).setattr("std_bias", value) }
    #[getter]
    fn estimated_std_bias(&self, py: Python<'_>) -> PyResult<PyObject> { Ok(self.inner.as_ref(py).getattr("estimated_std_bias")?.into()) }
    #[setter]
    fn set_estimated_std_bias(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> { self.inner.as_ref(py).setattr("estimated_std_bias", value) }
    #[getter]
    fn last_bias_time(&self, py: Python<'_>) -> PyResult<f64> { self.inner.as_ref(py).getattr("last_bias_time")?.extract() }
    #[setter]
    fn set_last_bias_time(&mut self, py: Python<'_>, value: f64) -> PyResult<()> { self.inner.as_ref(py).setattr("last_bias_time", value) }
    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<PyObject> { Ok(self.inner.as_ref(py).getattr("bounds")?.into()) }
    #[setter]
    fn set_bounds(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> { self.inner.as_ref(py).setattr("bounds", value) }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> { self.inner.as_ref(py).is_true() }
    fn copy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.as_ref(py).call_method0("copy")?.into();
        Ok(Py::new(py, Bias { inner })?.into_py(py))
    }
    #[pyo3(signature = (length, *, owner="sensor"))]
    fn broadcast_to(&self, py: Python<'_>, length: usize, owner: &str) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py); kwargs.set_item("owner", owner)?;
        let inner = self.inner.as_ref(py).call_method("broadcast_to", (length,), Some(kwargs))?.into();
        Ok(Py::new(py, Bias { inner })?.into_py(py))
    }
    #[pyo3(signature = (dt=1.0, *, form="full"))]
    fn process_covariance(&self, py: Python<'_>, dt: f64, form: &str) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py); kwargs.set_item("form", form)?;
        Ok(self.inner.as_ref(py).call_method("process_covariance", (dt,), Some(kwargs))?.into())
    }
    #[pyo3(signature = (*, form="full"))]
    fn process_psd(&self, py: Python<'_>, form: &str) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py); kwargs.set_item("form", form)?;
        Ok(self.inner.as_ref(py).call_method("process_psd", (), Some(kwargs))?.into())
    }
    #[pyo3(signature = (j2000, rng=None))]
    fn _update_bias(&self, py: Python<'_>, j2000: f64, rng: Option<PyObject>) -> PyResult<()> {
        match rng {
            Some(rng) => { self.inner.as_ref(py).call_method1("_update_bias", (j2000, rng))?; }
            None => { self.inner.as_ref(py).call_method1("_update_bias", (j2000,))?; }
        }
        Ok(())
    }
    #[pyo3(name = "get_bias")]
    fn current_bias(&self, py: Python<'_>, j2000: f64) -> PyResult<PyObject> { Ok(self.inner.as_ref(py).call_method1("get_bias", (j2000,))?.into()) }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, py: Python<'_>, value: PyObject, dt: f64, rng: Option<PyObject>) -> PyResult<PyObject> {
        match rng { Some(rng) => Ok(self.inner.as_ref(py).call_method1("apply", (value, dt, rng))?.into()), None => Ok(self.inner.as_ref(py).call_method1("apply", (value, dt))?.into()) }
    }
}
