use pyo3::prelude::*;

use super::adcs_subsystem::ADCSSubsystem as NativeADCSSubsystem;

#[pyclass(name = "ADCSSubsystem")]
pub struct ADCSSubsystem {
    inner: NativeADCSSubsystem<PyObject, PyObject>,
}

#[pymethods]
impl ADCSSubsystem {
    #[new]
    #[pyo3(signature = (reaction_wheels=None, magnetorquers=None))]
    fn new(reaction_wheels: Option<Vec<PyObject>>, magnetorquers: Option<Vec<PyObject>>) -> Self {
        Self {
            inner: NativeADCSSubsystem::new(
                reaction_wheels.unwrap_or_default(),
                magnetorquers.unwrap_or_default(),
            ),
        }
    }

    #[getter]
    fn reaction_wheels(&self, py: Python<'_>) -> PyObject {
        self.inner.reaction_wheels.clone().into_py(py)
    }

    #[setter]
    fn set_reaction_wheels(&mut self, reaction_wheels: Vec<PyObject>) {
        self.inner.reaction_wheels = reaction_wheels;
    }

    #[getter]
    fn magnetorquers(&self, py: Python<'_>) -> PyObject {
        self.inner.magnetorquers.clone().into_py(py)
    }

    #[setter]
    fn set_magnetorquers(&mut self, magnetorquers: Vec<PyObject>) {
        self.inner.magnetorquers = magnetorquers;
    }
}
