use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::attitude_mekf_py::parse_sensor_config;
use super::attitude_sequential_ukf::{self, AttitudeSequentialUKF};
use super::base::{AttitudeEstimator, Mat3, Mat6};
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

fn tuple3(py: Python<'_>, q: [f64; 4], w: [f64; 3], p: Mat6) -> PyResult<PyObject> {
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

#[pyclass(name = "AttitudeSequentialUKF", extends = PyAttitudeEstimator)]
pub struct PyAttitudeSequentialUKF {
    pub(crate) inner: AttitudeSequentialUKF,
}

#[pymethods]
impl PyAttitudeSequentialUKF {
    #[classattr]
    const COVARIANCE_SIZE: usize = AttitudeSequentialUKF::COVARIANCE_SIZE;

    #[new]
    #[pyo3(signature=(quaternion, rate, covariance, acceleration_psd, alpha=1.0, beta=2.0, kappa=0.0, record=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        quaternion: &PyAny,
        rate: &PyAny,
        covariance: &PyAny,
        acceleration_psd: &PyAny,
        alpha: f64,
        beta: f64,
        kappa: f64,
        record: Option<&PyAny>,
    ) -> PyResult<(Self, PyAttitudeEstimator)> {
        let q = vector(py, quaternion, "quaternion")?;
        let w = vector(py, rate, "rate")?;
        let p = matrix::<6, 6>(py, covariance, "covariance")?;
        let a = parse_psd(py, acceleration_psd)?;
        let record = match record {
            Some(value) => value.extract::<Vec<String>>()?,
            None => Vec::new(),
        };
        let base = AttitudeEstimator::new(q, w, p, a, record).map_err(PyValueError::new_err)?;
        let inner =
            AttitudeSequentialUKF::new(base.clone(), alpha, beta, kappa).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyAttitudeEstimator { inner: base }))
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }

    #[getter]
    fn ukf_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.ukf_params().to_vec())
    }
}

#[pyfunction]
#[pyo3(signature=(q, w, p, acceleration_psd, dt, alpha=1.0, beta=2.0, kappa=0.0))]
#[allow(clippy::too_many_arguments)]
fn ukf_predict(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<PyObject> {
    let state = attitude_sequential_ukf::predict(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        parse_psd(py, acceleration_psd)?,
        dt,
        alpha,
        beta,
        kappa,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
#[pyo3(signature=(q, w, p, acceleration_psd, dt, readings, sensor_config, references=None, alpha=1.0, beta=2.0, kappa=0.0))]
#[allow(clippy::too_many_arguments)]
fn ukf_step(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
    readings: &PyAny,
    sensor_config: &PyAny,
    references: Option<&PyAny>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<PyObject> {
    let reading: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (readings, "float64"))?
        .call_method0("ravel")?
        .extract()?;
    let state = attitude_sequential_ukf::step(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        parse_psd(py, acceleration_psd)?,
        dt,
        &reading,
        &parse_sensor_config(py, sensor_config, references)?,
        alpha,
        beta,
        kappa,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyAttitudeSequentialUKF>()?;
    module.add_function(wrap_pyfunction!(ukf_predict, module)?)?;
    module.add_function(wrap_pyfunction!(ukf_step, module)?)?;
    Ok(())
}
