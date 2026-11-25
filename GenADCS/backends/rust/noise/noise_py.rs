use super::noise::Noise as NativeNoise;
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
};

fn values(py: Python<'_>, x: &PyAny) -> PyResult<Vec<f64>> {
    py.import("numpy")?
        .call_method1("asarray", (x, "float64"))?
        .call_method0("ravel")?
        .extract()
}
fn result(py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
fn covariance_values(py: Python<'_>, x: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    py.import("numpy")?
        .call_method1("asarray", (x, "float64"))?
        .call_method0("tolist")?
        .extract()
}

#[pyclass(name = "Noise")]
pub struct Noise {
    inner: NativeNoise,
}
#[pymethods]
impl Noise {
    #[new]
    #[pyo3(signature=(covariance=None))]
    fn new(py: Python<'_>, covariance: Option<&PyAny>) -> PyResult<Self> {
        let c = covariance.map(|x| covariance_values(py, x)).transpose()?;
        Ok(Self {
            inner: NativeNoise::new(c).map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn covariance(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.covariance {
            Some(c) => Ok(py
                .import("numpy")?
                .call_method1("asarray", (c.clone(),))?
                .into()),
            None => Ok(py.None()),
        }
    }
    #[getter]
    fn cholesky_factor(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.cholesky_factor {
            Some(c) => Ok(py
                .import("numpy")?
                .call_method1("asarray", (c.clone(),))?
                .into()),
            None => Ok(py.None()),
        }
    }
    #[pyo3(signature=(value,dt,rng=None))]
    fn apply(
        &self,
        py: Python<'_>,
        value: &PyAny,
        dt: f64,
        rng: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let v = values(py, value)?;
        let output = if let Some(rng) = rng {
            let draw: Vec<f64> = rng.call_method1("standard_normal", (v.len(),))?.extract()?;
            self.inner.apply_with_draw(&v, &draw)
        } else {
            self.inner.apply(&v, dt)
        };
        result(
            py,
            output.map_err(|e| {
                if e.starts_with("configure") {
                    PyNotImplementedError::new_err(e)
                } else {
                    PyValueError::new_err(e)
                }
            })?,
        )
    }
}
