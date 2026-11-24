#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod _spherical_harmonics;
mod _spherical_harmonics_py;
use _spherical_harmonics::{
    calendar_jd, evaluate, gcrf_ecef, padded, parse_shc, tt_jd, Coefficients,
};
use _spherical_harmonics_py::{_schmidt_table, py_vec3, read_shc, vec3_from_py};
pub mod magnetic_field_model;
mod magnetic_field_model_py;
use magnetic_field_model_py::{magnetic_field_inertial_tesla, MagneticFieldModel, WMM};
#[path = "igrf14/igrf14.rs"]
pub mod igrf14;
pub use igrf14::Igrf14;
#[path = "igrf14/igrf14_py.rs"]
mod igrf14_py;
use igrf14_py::IGRF14;
#[path = "chaos8/chaos8.rs"]
pub mod chaos8;
pub use chaos8::Chaos8;
#[path = "chaos8/chaos8_py.rs"]
mod chaos8_py;
use chaos8_py::CHAOS8;
include!("native.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/environment/magnetic_field_model/test_magnetic_field.rs"]
mod tests;

#[pymodule]
fn _magnetic_field_model(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(magnetic_field_inertial_tesla, m)?)?;
    m.add_function(wrap_pyfunction!(read_shc, m)?)?;
    m.add_function(wrap_pyfunction!(_schmidt_table, m)?)?;
    m.add_class::<MagneticFieldModel>()?;
    m.add_class::<IGRF14>()?;
    m.add_class::<CHAOS8>()?;
    m.add_class::<WMM>()?;
    let _ = py;
    Ok(())
}
