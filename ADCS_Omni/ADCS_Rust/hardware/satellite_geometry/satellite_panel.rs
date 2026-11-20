#[pyclass(name = "SatellitePanel")]
#[derive(Clone)]
pub struct SatellitePanel {
    #[pyo3(get, set)]
    position: PyObject,
    #[pyo3(get, set)]
    orientation: PyObject,
    #[pyo3(get, set)]
    dimensions: PyObject,
    #[pyo3(get, set)]
    eta_s: f64,
    #[pyo3(get, set)]
    eta_d: f64,
    #[pyo3(get, set)]
    eta_a: f64,
    #[pyo3(get, set)]
    cd: f64,
}

#[pymethods]
impl SatellitePanel {
    #[new]
    #[pyo3(signature = (position, orientation, dimensions, eta_s=0.0, eta_d=0.0, eta_a=1.0, cd=2.2))]
    fn new(
        position: PyObject,
        orientation: PyObject,
        dimensions: PyObject,
        eta_s: f64,
        eta_d: f64,
        eta_a: f64,
        cd: f64,
    ) -> Self {
        Self {
            position,
            orientation,
            dimensions,
            eta_s,
            eta_d,
            eta_a,
            cd,
        }
    }
}
