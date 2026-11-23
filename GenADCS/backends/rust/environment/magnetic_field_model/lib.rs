#![allow(non_local_definitions)]
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

include!("_spherical_harmonics.rs");
include!("magnetic_field_model.rs");
include!("igrf14/igrf14.rs");
include!("chaos8/chaos8.rs");
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
