use pyo3::prelude::*;

use crate::dynamics_config::DynamicsConfigCore;

#[pyclass]
pub struct DynamicsConfig {
    pub inner: DynamicsConfigCore,
    pub forces: Vec<PyObject>,
    pub torques: Vec<PyObject>,
}

fn model_kinds(py: Python<'_>, models: &[PyObject]) -> PyResult<Vec<String>> {
    models
        .iter()
        .map(|model| {
            model
                .as_ref(py)
                .getattr("__class__")?
                .getattr("__name__")?
                .extract()
        })
        .collect()
}

#[pymethods]
impl DynamicsConfig {
    #[new]
    #[pyo3(signature = (forces=None, torques=None))]
    fn new(
        py: Python<'_>,
        forces: Option<Vec<PyObject>>,
        torques: Option<Vec<PyObject>>,
    ) -> Self {
        let forces = forces.unwrap_or_default();
        let torques = torques.unwrap_or_default();
        let force_model_kinds = model_kinds(py, &forces).unwrap_or_default();
        let torque_model_kinds = model_kinds(py, &torques).unwrap_or_default();
        let inner = DynamicsConfigCore::new(force_model_kinds, torque_model_kinds);
        Self { inner, forces, torques }
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
