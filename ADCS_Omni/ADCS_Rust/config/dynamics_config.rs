#[pyclass]
pub struct DynamicsConfig {
    forces: Vec<PyObject>,
    torques: Vec<PyObject>,
}

#[pymethods]
impl DynamicsConfig {
    #[new]
    #[pyo3(signature = (forces=None, torques=None))]
    fn new(forces: Option<Vec<PyObject>>, torques: Option<Vec<PyObject>>) -> Self {
        Self {
            forces: forces.unwrap_or_default(),
            torques: torques.unwrap_or_default(),
        }
    }

    #[getter]
    fn forces(&self, py: Python<'_>) -> PyObject {
        self.forces.clone().into_py(py)
    }

    #[setter]
    fn set_forces(&mut self, forces: Vec<PyObject>) {
        self.forces = forces;
    }

    #[getter]
    fn torques(&self, py: Python<'_>) -> PyObject {
        self.torques.clone().into_py(py)
    }

    #[setter]
    fn set_torques(&mut self, torques: Vec<PyObject>) {
        self.torques = torques;
    }
}
