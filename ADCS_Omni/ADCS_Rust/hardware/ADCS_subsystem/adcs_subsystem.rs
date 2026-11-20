#[pyclass(name = "ADCSSubsystem")]
pub struct ADCSSubsystem {
    reaction_wheels: Vec<PyObject>,
    magnetorquers: Vec<PyObject>,
}

#[pymethods]
impl ADCSSubsystem {
    #[new]
    #[pyo3(signature = (reaction_wheels=None, magnetorquers=None))]
    fn new(reaction_wheels: Option<Vec<PyObject>>, magnetorquers: Option<Vec<PyObject>>) -> Self {
        Self {
            reaction_wheels: reaction_wheels.unwrap_or_default(),
            magnetorquers: magnetorquers.unwrap_or_default(),
        }
    }

    #[getter]
    fn reaction_wheels(&self, py: Python<'_>) -> PyObject {
        self.reaction_wheels.clone().into_py(py)
    }

    #[setter]
    fn set_reaction_wheels(&mut self, reaction_wheels: Vec<PyObject>) {
        self.reaction_wheels = reaction_wheels;
    }

    #[getter]
    fn magnetorquers(&self, py: Python<'_>) -> PyObject {
        self.magnetorquers.clone().into_py(py)
    }

    #[setter]
    fn set_magnetorquers(&mut self, magnetorquers: Vec<PyObject>) {
        self.magnetorquers = magnetorquers;
    }
}
