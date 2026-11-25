use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::Sensor;

pub(super) fn vector<const N: usize>(
    py: Python<'_>,
    value: &PyAny,
    label: &str,
) -> PyResult<[f64; N]> {
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
        .map_err(|_| PyValueError::new_err(format!("{label} must have {N} components")))
}

pub(super) fn matrix3(py: Python<'_>, value: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    if shape != [3, 3] {
        return Err(PyValueError::new_err("noise must be a 3x3 covariance"));
    }
    let matrix: Vec<Vec<f64>> = array.call_method0("tolist")?.extract()?;
    if matrix.len() != 3 || matrix.iter().any(|row| row.len() != 3) {
        return Err(PyValueError::new_err("noise must be a 3x3 covariance"));
    }
    Ok(matrix)
}

pub(super) fn array(py: Python<'_>, values: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (values,))?
        .into())
}

#[pyclass(name = "Sensor")]
pub struct PySensor {
    inner: Sensor,
}

#[pymethods]
impl PySensor {
    #[new]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        misalignment: &PyAny,
        saturation_value: f64,
        noise: &PyAny,
        bias: &PyAny,
    ) -> PyResult<Self> {
        let position = vector::<3>(py, position, "position")?;
        let orientation = vector::<4>(py, orientation, "orientation")?;
        let misalignment = vector::<4>(py, misalignment, "misalignment")?;
        let bias = vector::<3>(py, bias, "bias")?;
        let noise = matrix3(py, noise)?;
        let inner = Sensor::new(
            &position,
            &orientation,
            &misalignment,
            saturation_value,
            &noise,
            &bias,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.position.to_vec())
    }
    #[setter]
    fn set_position(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.position = vector(py, value, "position")?;
        Ok(())
    }

    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.orientation.to_vec())
    }
    #[setter]
    fn set_orientation(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.orientation = vector(py, value, "orientation")?;
        Ok(())
    }

    #[getter]
    fn misalignment(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.misalignment.to_vec())
    }
    #[setter]
    fn set_misalignment(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.misalignment = vector(py, value, "misalignment")?;
        Ok(())
    }

    #[getter]
    fn saturation_value(&self) -> f64 {
        self.inner.saturation_value
    }
    #[setter]
    fn set_saturation_value(&mut self, value: f64) {
        self.inner.saturation_value = value;
    }

    #[getter]
    fn noise(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.noise.to_vec())
    }
    #[setter]
    fn set_noise(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let rows = matrix3(py, value)?;
        self.inner.noise = std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j]));
        Ok(())
    }

    #[getter]
    fn bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.bias.to_vec())
    }
    #[setter]
    fn set_bias(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.inner.bias = vector(py, value, "bias")?;
        Ok(())
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[setter]
    fn set_enabled(&mut self, value: bool) {
        self.inner.enabled = value;
    }

    fn availability(&self) -> bool {
        self.inner.availability()
    }
}
