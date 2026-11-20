use crate::state_types::{Quaternion, Vector3};
use crate::state_validation::finite;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct AttitudeState {
    pub orientation: Quaternion,
    pub rates: Vector3,
}

#[pymethods]
impl AttitudeState {
    #[new]
    fn new(py: Python<'_>, orientation: &PyAny, rates: &PyAny) -> PyResult<Self> {
        let orientation = quaternion(py, orientation)?;
        let rates = vector3(py, rates, "rates")?;
        Ok(Self { orientation, rates })
    }

    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.orientation.iter().copied())
    }
    #[setter]
    fn set_orientation(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.orientation = quaternion(py, value)?;
        Ok(())
    }
    #[getter]
    fn rates(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.rates.iter().copied())
    }
    #[setter]
    fn set_rates(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.rates = vector3(py, value, "rates")?;
        Ok(())
    }
    #[getter]
    fn quaternion_array(&self, py: Python<'_>) -> PyResult<PyObject> {
        let norm = self.orientation.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(PyValueError::new_err(
                "orientation quaternion must be finite and nonzero",
            ));
        }
        array(py, self.orientation.iter().map(|v| v / norm))
    }
    #[staticmethod]
    pub fn from_python(py: Python<'_>, state: &PyAny) -> PyResult<Self> {
        let orientation = state.getattr("orientation")?;
        let orientation = if orientation.hasattr("hamilton_components")? {
            orientation.getattr("hamilton_components")?
        } else {
            orientation
        };
        Self::new(py, orientation, state.getattr("rates")?)
    }
    pub fn to_python(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py
            .import("ADCS_Omni.ADCS_Python.state")?
            .getattr("AttitudeState")?;
        Ok(cls
            .call1((
                array(py, self.orientation.iter().copied())?,
                array(py, self.rates.iter().copied())?,
            ))?
            .into())
    }
}

pub(crate) fn values(
    py: Python<'_>,
    value: &PyAny,
    name: &str,
    size: Option<usize>,
) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let array = np
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be a finite 1D array"
        )));
    }
    let result: Vec<f64> = array.call_method0("tolist")?.extract()?;
    finite(&result, name, size)?;
    Ok(result)
}

pub(crate) fn vector3(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<Vector3> {
    values(py, value, name, Some(3))?.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("{name} must be a finite 3-vector"))
    })
}

pub(crate) fn quaternion(py: Python<'_>, value: &PyAny) -> PyResult<Quaternion> {
    let value = if value.hasattr("hamilton_components")? {
        value.getattr("hamilton_components")?
    } else {
        value
    };
    let result: Quaternion = values(py, value, "orientation", Some(4))?
        .try_into()
        .map_err(|_| PyValueError::new_err("orientation must be a finite 4-vector"))?;
    if result.iter().map(|v| v * v).sum::<f64>() == 0.0 {
        return Err(PyValueError::new_err(
            "orientation quaternion must be finite and nonzero",
        ));
    }
    Ok(result)
}

pub(crate) fn array<T: IntoIterator<Item = f64>>(py: Python<'_>, values: T) -> PyResult<PyObject> {
    Ok(py
        .import("numpy")?
        .call_method1("array", (values.into_iter().collect::<Vec<_>>(),))?
        .into())
}
