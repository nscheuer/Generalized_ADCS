#[pyclass(name = "SatelliteGeometry")]
pub struct SatelliteGeometry {
    #[pyo3(get, set)]
    mass: f64,
    #[pyo3(get, set)]
    inertia: PyObject,
    panels: Vec<PyObject>,
    center_of_mass: PyObject,
}

#[pymethods]
impl SatelliteGeometry {
    #[new]
    #[pyo3(signature = (mass, inertia, panels=None, center_of_mass=None))]
    fn new(
        py: Python<'_>,
        mass: f64,
        inertia: PyObject,
        panels: Option<Vec<PyObject>>,
        center_of_mass: Option<PyObject>,
    ) -> Self {
        Self {
            mass,
            inertia,
            panels: panels.unwrap_or_default(),
            center_of_mass: center_of_mass.unwrap_or_else(|| (0.0, 0.0, 0.0).into_py(py)),
        }
    }

    #[getter]
    fn panels(&self, py: Python<'_>) -> PyObject {
        self.panels.clone().into_py(py)
    }

    #[setter]
    fn set_panels(&mut self, panels: Vec<PyObject>) {
        self.panels = panels;
    }

    #[getter]
    fn center_of_mass(&self, py: Python<'_>) -> PyObject {
        self.center_of_mass.clone_ref(py)
    }

    #[setter]
    fn set_center_of_mass(&mut self, center_of_mass: PyObject) {
        self.center_of_mass = center_of_mass;
    }
}
