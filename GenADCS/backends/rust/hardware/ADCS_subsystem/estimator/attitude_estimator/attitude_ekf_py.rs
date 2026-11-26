use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::attitude_ekf::{self, AttitudeEKF, Mat7};
use super::attitude_mekf_py::parse_sensor_config;
use super::base::{AttitudeEstimator, Mat3};
use super::base_py::PyAttitudeEstimator;

fn vector<const N: usize>(py: Python<'_>, value: &PyAny, label: &str) -> PyResult<[f64; N]> {
    let arr = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape != [N] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({N},)"
        )));
    }
    arr.call_method0("tolist")?
        .extract::<Vec<f64>>()?
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{label} must have shape ({N},)")))
}

fn matrix<const R: usize, const C: usize>(
    py: Python<'_>,
    value: &PyAny,
    label: &str,
) -> PyResult<[[f64; C]; R]> {
    let arr = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape != [R, C] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({R}, {C})"
        )));
    }
    let rows: Vec<Vec<f64>> = arr.call_method0("tolist")?.extract()?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j])))
}

fn out(py: Python<'_>, value: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .into())
}

fn tuple3(py: Python<'_>, q: [f64; 4], w: [f64; 3], p: Mat7) -> PyResult<PyObject> {
    let a = out(py, q.to_vec())?;
    let b = out(py, w.to_vec())?;
    let c = out(py, p.to_vec())?;
    Ok(pyo3::types::PyTuple::new(py, [a, b, c]).into())
}

fn parse_psd(py: Python<'_>, acceleration_psd: &PyAny) -> PyResult<Mat3> {
    let arr = py
        .import("numpy")?
        .call_method1("asarray", (acceleration_psd, "float64"))?;
    if arr.getattr("ndim")?.extract::<usize>()? == 0 {
        let x: f64 = arr.extract()?;
        Ok([[x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, x]])
    } else {
        matrix::<3, 3>(py, acceleration_psd, "acceleration_psd")
    }
}

#[pyclass(name = "AttitudeEKF", extends = PyAttitudeEstimator)]
pub struct PyAttitudeEKF {
    pub(crate) inner: AttitudeEKF,
}

#[pymethods]
impl PyAttitudeEKF {
    #[new]
    #[pyo3(signature=(quaternion, rate, covariance, acceleration_psd, record=None))]
    fn new(
        py: Python<'_>,
        quaternion: &PyAny,
        rate: &PyAny,
        covariance: &PyAny,
        acceleration_psd: &PyAny,
        record: Option<&PyAny>,
    ) -> PyResult<(Self, PyAttitudeEstimator)> {
        let q = vector(py, quaternion, "quaternion")?;
        let w = vector(py, rate, "rate")?;
        let p = matrix::<7, 7>(py, covariance, "covariance")?;
        let a = parse_psd(py, acceleration_psd)?;
        let record = match record {
            Some(value) => value.extract::<Vec<String>>()?,
            None => Vec::new(),
        };
        let inner = AttitudeEKF::new(q, w, p, a, record.clone()).map_err(PyValueError::new_err)?;
        let base = AttitudeEstimator::new(q, w, [[0.0; 6]; 6], a, record)
            .map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyAttitudeEstimator { inner: base }))
    }

    #[getter]
    fn quaternion(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.quaternion.to_vec())
    }
    #[setter]
    fn set_quaternion(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.quaternion = vector(py, value, "quaternion")?;
        Ok(())
    }
    #[getter]
    fn rate(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.rate.to_vec())
    }
    #[setter]
    fn set_rate(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.rate = vector(py, value, "rate")?;
        Ok(())
    }
    #[getter]
    fn covariance(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.covariance.to_vec())
    }
    #[setter]
    fn set_covariance(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.covariance = matrix::<7, 7>(py, value, "covariance")?;
        Ok(())
    }
    #[getter]
    fn acceleration_psd(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.acceleration_psd.to_vec())
    }
    #[setter]
    fn set_acceleration_psd(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.acceleration_psd = parse_psd(py, value)?;
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
            out(py, q.to_vec())?,
            out(py, w.to_vec())?,
            out(py, p.to_vec())?,
            out(py, a.to_vec())?,
        ];
        Ok(pyo3::types::PyTuple::new(py, values).into())
    }
}

#[pyfunction]
fn ekf_predict(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
) -> PyResult<PyObject> {
    let state = attitude_ekf::predict(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<7, 7>(py, p, "covariance")?,
        parse_psd(py, acceleration_psd)?,
        dt,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
fn ekf_correct(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    residual: &PyAny,
    h: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_ekf::correct(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<7, 7>(py, p, "covariance")?,
        vector(py, residual, "residual")?,
        matrix::<3, 7>(py, h, "measurement matrix")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
fn ekf_correct_gyro(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    measured_rate: &PyAny,
    mounting: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_ekf::correct_gyro(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<7, 7>(py, p, "covariance")?,
        vector(py, measured_rate, "measured_rate")?,
        matrix::<3, 3>(py, mounting, "mounting")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
fn ekf_correct_star_tracker(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    measured_quaternion: &PyAny,
    mounting_quaternion: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_ekf::correct_star_tracker(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<7, 7>(py, p, "covariance")?,
        vector(py, measured_quaternion, "measured quaternion")?,
        vector(py, mounting_quaternion, "mounting quaternion")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
#[pyo3(signature=(q, w, p, acceleration_psd, dt, readings, sensor_config, references=None))]
fn ekf_step(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
    readings: &PyAny,
    sensor_config: &PyAny,
    references: Option<&PyAny>,
) -> PyResult<PyObject> {
    let reading: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (readings, "float64"))?
        .call_method0("ravel")?
        .extract()?;
    let state = attitude_ekf::step(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<7, 7>(py, p, "covariance")?,
        parse_psd(py, acceleration_psd)?,
        dt,
        &reading,
        &parse_sensor_config(py, sensor_config, references)?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyAttitudeEKF>()?;
    module.add_function(wrap_pyfunction!(ekf_predict, module)?)?;
    module.add_function(wrap_pyfunction!(ekf_correct, module)?)?;
    module.add_function(wrap_pyfunction!(ekf_correct_gyro, module)?)?;
    module.add_function(wrap_pyfunction!(ekf_correct_star_tracker, module)?)?;
    module.add_function(wrap_pyfunction!(ekf_step, module)?)?;
    Ok(())
}
