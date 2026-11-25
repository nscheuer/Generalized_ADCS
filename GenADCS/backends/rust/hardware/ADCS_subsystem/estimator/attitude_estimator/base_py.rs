use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::base::{AttitudeEstimator, Mat3};

fn vector<const N: usize>(py: Python<'_>, value: &PyAny, label: &str) -> PyResult<[f64; N]> {
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    if shape != [N] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({N},)"
        )));
    }
    let values: Vec<f64> = array.call_method0("tolist")?.extract()?;
    values
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{label} must have shape ({N},)")))
}
fn matrix<const N: usize>(py: Python<'_>, value: &PyAny, label: &str) -> PyResult<[[f64; N]; N]> {
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    if shape != [N, N] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({N}, {N})"
        )));
    }
    let rows: Vec<Vec<f64>> = array.call_method0("tolist")?.extract()?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j])))
}
fn as_array(py: Python<'_>, values: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (values,))?
        .into())
}

#[pyclass(name = "AttitudeEstimator", subclass)]
pub struct PyAttitudeEstimator {
    pub(crate) inner: AttitudeEstimator,
}

#[pymethods]
impl PyAttitudeEstimator {
    #[new]
    #[pyo3(signature=(quaternion,rate,covariance,acceleration_psd,record=None))]
    fn new(
        py: Python<'_>,
        quaternion: &PyAny,
        rate: &PyAny,
        covariance: &PyAny,
        acceleration_psd: &PyAny,
        record: Option<&PyAny>,
    ) -> PyResult<Self> {
        let q = vector(py, quaternion, "quaternion")?;
        let w = vector(py, rate, "rate")?;
        let p = matrix::<6>(py, covariance, "covariance")?;
        let acc_array = py
            .import("numpy")?
            .call_method1("asarray", (acceleration_psd, "float64"))?;
        let acceleration: Mat3 = if acc_array.getattr("ndim")?.extract::<usize>()? == 0 {
            let scalar: f64 = acc_array.extract()?;
            [[scalar, 0., 0.], [0., scalar, 0.], [0., 0., scalar]]
        } else {
            matrix::<3>(py, acceleration_psd, "acceleration_psd")?
        };
        let record = match record {
            Some(r) => r.extract::<Vec<String>>()?,
            None => Vec::new(),
        };
        let inner =
            AttitudeEstimator::new(q, w, p, acceleration, record).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn quaternion(&self, py: Python<'_>) -> PyResult<PyObject> {
        as_array(py, self.inner.quaternion.to_vec())
    }
    #[setter]
    fn set_quaternion(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.quaternion = vector(py, value, "quaternion")?;
        Ok(())
    }
    #[getter]
    fn rate(&self, py: Python<'_>) -> PyResult<PyObject> {
        as_array(py, self.inner.rate.to_vec())
    }
    #[setter]
    fn set_rate(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.rate = vector(py, value, "rate")?;
        Ok(())
    }
    #[getter]
    fn covariance(&self, py: Python<'_>) -> PyResult<PyObject> {
        as_array(py, self.inner.covariance.to_vec())
    }
    #[setter]
    fn set_covariance(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.covariance = matrix(py, value, "covariance")?;
        Ok(())
    }
    #[getter]
    fn acceleration_psd(&self, py: Python<'_>) -> PyResult<PyObject> {
        as_array(py, self.inner.acceleration_psd.to_vec())
    }
    #[setter]
    fn set_acceleration_psd(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let arr = py
            .import("numpy")?
            .call_method1("asarray", (value, "float64"))?;
        self.inner.acceleration_psd = if arr.getattr("ndim")?.extract::<usize>()? == 0 {
            let x: f64 = arr.extract()?;
            [[x, 0., 0.], [0., x, 0.], [0., 0., x]]
        } else {
            matrix(py, value, "acceleration_psd")?
        };
        Ok(())
    }
    #[getter]
    fn record(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(py
            .import("builtins")?
            .getattr("frozenset")?
            .call1((self.inner.record.clone(),))?
            .into())
    }
    #[setter]
    fn set_record(&mut self, value: &PyAny) -> PyResult<()> {
        self.inner.record = value.extract::<Vec<String>>()?;
        Ok(())
    }
    fn prepare(&self, py: Python<'_>) -> PyResult<PyObject> {
        let (q, w, p, a) = self.inner.prepare();
        let values = [
            as_array(py, q.to_vec())?,
            as_array(py, w.to_vec())?,
            as_array(py, p.to_vec())?,
            as_array(py, a.to_vec())?,
        ];
        Ok(pyo3::types::PyTuple::new(py, values).into())
    }
}
