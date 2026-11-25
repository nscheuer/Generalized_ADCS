use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::attitude_mekf::{self, AttitudeMEKF, SensorConfig};
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

fn rows(py: Python<'_>, value: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    py.import("numpy")?
        .call_method1("asarray", (value, "float64"))?
        .call_method0("tolist")?
        .extract()
}
fn parse_sensor_config(py: Python<'_>, value: &PyAny) -> PyResult<SensorConfig> {
    let item = |index| value.get_item(index);
    let code: Vec<i64> = item(0)?.call_method0("tolist")?.extract()?;
    let orientation = rows(py, item(1)?)?;
    let misalignment = rows(py, item(2)?)?;
    let bias = rows(py, item(3)?)?;
    let cholesky = py
        .import("numpy")?
        .call_method1("asarray", (item(4)?, "float64"))?
        .call_method0("tolist")?
        .extract::<Vec<Vec<Vec<f64>>>>()?;
    let enabled: Vec<bool> = item(8)?.call_method0("tolist")?.extract()?;
    let offsets: Vec<usize> = item(9)?.call_method0("tolist")?.extract()?;
    let count = code.len();
    if [
        orientation.len(),
        misalignment.len(),
        bias.len(),
        cholesky.len(),
        enabled.len(),
        offsets.len(),
    ]
    .iter()
    .any(|n| *n != count)
    {
        return Err(PyValueError::new_err(
            "sensor configuration rows have inconsistent lengths",
        ));
    }
    let orientation = orientation
        .into_iter()
        .map(|r| {
            r.try_into()
                .map_err(|_| PyValueError::new_err("sensor orientation must have four components"))
        })
        .collect::<PyResult<Vec<[f64; 4]>>>()?;
    let misalignment = misalignment
        .into_iter()
        .map(|r| {
            r.try_into()
                .map_err(|_| PyValueError::new_err("sensor misalignment must have four components"))
        })
        .collect::<PyResult<Vec<[f64; 4]>>>()?;
    let bias = bias
        .into_iter()
        .map(|r| {
            if r.len() != 6 {
                return Err(PyValueError::new_err(
                    "sensor bias rows must have six components",
                ));
            }
            Ok(std::array::from_fn(|i| r[i]))
        })
        .collect::<PyResult<Vec<[f64; 6]>>>()?;
    let noise_cholesky = cholesky
        .into_iter()
        .map(|m| {
            if m.len() != 6 || m.iter().any(|r| r.len() != 6) {
                return Err(PyValueError::new_err("sensor Cholesky rows must be 6x6"));
            }
            Ok(std::array::from_fn(|i| std::array::from_fn(|j| m[i][j])))
        })
        .collect::<PyResult<Vec<[[f64; 6]; 6]>>>()?;
    Ok(SensorConfig {
        code,
        orientation,
        misalignment,
        bias,
        noise_cholesky,
        enabled,
        offsets,
    })
}

#[pyclass(name="AttitudeMEKF",extends=PyAttitudeEstimator)]
pub struct PyAttitudeMEKF {
    pub(crate) _inner: AttitudeMEKF,
}

#[pymethods]
impl PyAttitudeMEKF {
    #[new]
    #[pyo3(signature=(quaternion,rate,covariance,acceleration_psd,record=None))]
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
        let p = matrix::<6, 6>(py, covariance, "covariance")?;
        let acceleration_array = py
            .import("numpy")?
            .call_method1("asarray", (acceleration_psd, "float64"))?;
        let a: Mat3 = if acceleration_array.getattr("ndim")?.extract::<usize>()? == 0 {
            let x: f64 = acceleration_array.extract()?;
            [[x, 0., 0.], [0., x, 0.], [0., 0., x]]
        } else {
            matrix(py, acceleration_psd, "acceleration_psd")?
        };
        let record = match record {
            Some(value) => value.extract::<Vec<String>>()?,
            None => Vec::new(),
        };
        let base = AttitudeEstimator::new(q, w, p, a, record).map_err(PyValueError::new_err)?;
        Ok((
            Self {
                _inner: AttitudeMEKF::new(base.clone()),
            },
            PyAttitudeEstimator { inner: base },
        ))
    }
}

#[pyfunction]
fn predict(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
) -> PyResult<PyObject> {
    let state = attitude_mekf::predict(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        matrix::<3, 3>(py, acceleration_psd, "acceleration_psd")?,
        dt,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}
#[pyfunction]
fn correct(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    residual: &PyAny,
    h: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_mekf::correct(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        vector(py, residual, "residual")?,
        matrix::<3, 6>(py, h, "measurement matrix")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}
#[pyfunction]
fn correct_gyro(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    measured_rate: &PyAny,
    mounting: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_mekf::correct_gyro(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        vector(py, measured_rate, "measured_rate")?,
        matrix::<3, 3>(py, mounting, "mounting")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}
#[pyfunction]
fn correct_star_tracker(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    measured_quaternion: &PyAny,
    mounting_quaternion: &PyAny,
    covariance: &PyAny,
) -> PyResult<PyObject> {
    let state = attitude_mekf::correct_star_tracker(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix::<6, 6>(py, p, "covariance")?,
        vector(py, measured_quaternion, "measured quaternion")?,
        vector(py, mounting_quaternion, "mounting quaternion")?,
        matrix::<3, 3>(py, covariance, "measurement covariance")?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

#[pyfunction]
fn step(
    py: Python<'_>,
    q: &PyAny,
    w: &PyAny,
    p: &PyAny,
    acceleration_psd: &PyAny,
    dt: f64,
    readings: &PyAny,
    sensor_config: &PyAny,
) -> PyResult<PyObject> {
    let reading: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (readings, "float64"))?
        .call_method0("ravel")?
        .extract()?;
    let state = attitude_mekf::step(
        vector(py, q, "quaternion")?,
        vector(py, w, "rate")?,
        matrix(py, p, "covariance")?,
        matrix(py, acceleration_psd, "acceleration_psd")?,
        dt,
        &reading,
        &parse_sensor_config(py, sensor_config)?,
    )
    .map_err(PyValueError::new_err)?;
    tuple3(py, state.0, state.1, state.2)
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyAttitudeEstimator>()?;
    module.add_class::<PyAttitudeMEKF>()?;
    module.add_function(wrap_pyfunction!(predict, module)?)?;
    module.add_function(wrap_pyfunction!(correct, module)?)?;
    module.add_function(wrap_pyfunction!(correct_gyro, module)?)?;
    module.add_function(wrap_pyfunction!(correct_star_tracker, module)?)?;
    module.add_function(wrap_pyfunction!(step, module)?)?;
    Ok(())
}
