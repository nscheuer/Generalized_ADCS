use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "PlanetaryBody", subclass)]
pub struct PlanetaryBodyPy {
    pub(super) body: super::planetary_body::PlanetaryBody,
}

#[pymethods]
impl PlanetaryBodyPy {
    #[new]
    fn new(name: String, gravitational_parameter: f64, reference_radius: f64) -> PyResult<Self> {
        Ok(Self {
            body: super::planetary_body::PlanetaryBody::new(
                name,
                gravitational_parameter,
                reference_radius,
            )
            .map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        self.body.name()
    }

    #[setter]
    fn set_name(&mut self, value: String) -> PyResult<()> {
        self.body.set_name(value).map_err(PyValueError::new_err)
    }

    #[getter]
    fn gravitational_parameter(&self) -> f64 {
        self.body.gravitational_parameter()
    }

    #[setter]
    fn set_gravitational_parameter(&mut self, value: f64) -> PyResult<()> {
        self.body
            .set_gravitational_parameter(value)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn reference_radius(&self) -> f64 {
        self.body.reference_radius()
    }

    #[setter]
    fn set_reference_radius(&mut self, value: f64) -> PyResult<()> {
        self.body
            .set_reference_radius(value)
            .map_err(PyValueError::new_err)
    }

    fn __richcmp__(&self, other: PyRef<'_, PlanetaryBodyPy>, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.body == other.body,
            CompareOp::Ne => self.body != other.body,
            _ => false,
        }
    }
}
