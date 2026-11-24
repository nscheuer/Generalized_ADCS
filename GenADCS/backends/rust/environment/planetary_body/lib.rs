#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod earth;
mod earth_py;
pub mod planetary_body;
mod planetary_body_py;

#[pymodule]
fn _planetary_body(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<planetary_body_py::PlanetaryBodyPy>()?;
    m.add_class::<earth_py::EarthPy>()?;
    Ok(())
}
