#[pyclass(subclass)]
pub struct PlanetaryBody {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    gravitational_parameter: f64,
    #[pyo3(get, set)]
    reference_radius: f64,
}

#[pymethods]
impl PlanetaryBody {
    #[new]
    fn new(name: String, gravitational_parameter: f64, reference_radius: f64) -> Self {
        Self {
            name,
            gravitational_parameter,
            reference_radius,
        }
    }

    fn __richcmp__(
        &self,
        other: PyRef<'_, PlanetaryBody>,
        op: pyo3::class::basic::CompareOp,
    ) -> bool {
        let equal = self.name == other.name
            && self.gravitational_parameter == other.gravitational_parameter
            && self.reference_radius == other.reference_radius;
        match op {
            pyo3::class::basic::CompareOp::Eq => equal,
            pyo3::class::basic::CompareOp::Ne => !equal,
            _ => false,
        }
    }
}
