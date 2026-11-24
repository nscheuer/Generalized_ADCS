#![allow(non_local_definitions)]
use numpy::ToPyArray;
use pyo3::prelude::*;

include!("ephemeris_model.rs");
include!("ephemeris_model_py.rs");
include!("de421/de421.rs");
include!("de421/de421_py.rs");
include!("native.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/environment/ephemeris/test_de421.rs"]
mod tests;

#[pymodule]
fn _ephemeris(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("BODY_SUN", 0)?;
    m.add("BODY_EARTH", 1)?;
    m.add("BODY_MOON", 2)?;
    m.add_function(wrap_pyfunction!(eclipse_state_py, m)?)?;
    m.add_function(wrap_pyfunction!(ephemeris_position, m)?)?;
    m.add_function(wrap_pyfunction!(ephemeris_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(planetary_position, m)?)?;
    m.add_function(wrap_pyfunction!(planetary_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(planetary_state, m)?)?;
    m.add_class::<EphemerisModel>()?;
    m.add_class::<DE421>()?;
    Ok(())
}
