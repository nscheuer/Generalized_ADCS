use crate::attitude_state::{array, values};
use pyo3::prelude::*;

#[pyclass(subclass)]
pub struct InternalState {
    pub wheel_momentum: Vec<f64>,
}

#[pymethods]
impl InternalState {
    #[new]
    #[pyo3(signature = (wheel_momentum=None))]
    fn new(py: Python<'_>, wheel_momentum: Option<&PyAny>) -> PyResult<Self> {
        let wheel_momentum = wheel_momentum
            .map(|v| values(py, v, "wheel_momentum", None))
            .transpose()?
            .unwrap_or_default();
        Ok(Self { wheel_momentum })
    }
    #[getter]
    fn wheel_momentum(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.wheel_momentum.clone())
    }
    #[setter]
    fn set_wheel_momentum(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        self.wheel_momentum = values(py, value, "wheel_momentum", None)?;
        Ok(())
    }
    #[staticmethod]
    pub fn from_python(py: Python<'_>, state: &PyAny) -> PyResult<Self> {
        Self::new(py, Some(state.getattr("wheel_momentum")?))
    }
    pub fn to_python(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py
            .import("ADCS_Omni.ADCS_Python.state")?
            .getattr("InternalState")?;
        Ok(cls
            .call1((array(py, self.wheel_momentum.clone())?,))?
            .into())
    }
}

#[pyclass(extends=InternalState)]
pub struct WheelMomentumState;

#[pymethods]
impl WheelMomentumState {
    #[new]
    #[pyo3(signature = (subsystem=None, *, wheel_momentum=None))]
    fn new(
        py: Python<'_>,
        subsystem: Option<&PyAny>,
        wheel_momentum: Option<&PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        if subsystem.is_some() && wheel_momentum.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "provide subsystem or wheel_momentum, not both",
            ));
        }
        let values = if let Some(source) = subsystem {
            let mut result = Vec::new();
            for wheel in source.getattr("reaction_wheels")?.iter()? {
                result.push(wheel?.getattr("h")?.extract::<f64>()?);
            }
            result
        } else if let Some(momentum) = wheel_momentum {
            values(py, momentum, "wheel_momentum", None)?
        } else {
            Vec::new()
        };
        Ok(PyClassInitializer::from(InternalState {
            wheel_momentum: values,
        })
        .add_subclass(Self))
    }
    #[staticmethod]
    fn from_python(py: Python<'_>, state: &PyAny) -> PyResult<Py<Self>> {
        Py::new(
            py,
            PyClassInitializer::from(InternalState {
                wheel_momentum: values(
                    py,
                    state.getattr("wheel_momentum")?,
                    "wheel_momentum",
                    None,
                )?,
            })
            .add_subclass(Self),
        )
    }
    fn to_python(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        let base = slf.into_super();
        let cls = py
            .import("ADCS_Omni.ADCS_Python.state")?
            .getattr("WheelMomentumState")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("wheel_momentum", base.wheel_momentum.clone())?;
        Ok(cls.call((), Some(kwargs))?.into())
    }
}
