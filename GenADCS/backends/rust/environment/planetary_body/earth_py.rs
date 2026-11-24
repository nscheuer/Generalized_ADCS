use super::planetary_body_py::PlanetaryBodyPy;
use pyo3::prelude::*;

#[pyclass(name = "Earth", extends = PlanetaryBodyPy)]
pub struct EarthPy;

#[pymethods]
impl EarthPy {
    #[new]
    fn new() -> (Self, PlanetaryBodyPy) {
        (
            Self,
            PlanetaryBodyPy {
                body: super::earth::Earth::planetary_body(),
            },
        )
    }
}
