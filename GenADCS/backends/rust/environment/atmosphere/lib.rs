#![allow(non_local_definitions)]
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

include!("NRMLSISE00/_nrlmsise00_core.rs");
include!("NRMLSISE00/nrlmsise00.rs");
include!("NRMLSISE00/nrlmsise00_py.rs");
include!("atmospheric_model.rs");
include!("atmospheric_model_py.rs");
include!("native.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/environment/atmosphere/test_atmosphere.rs"]
mod tests;

#[pymodule]
fn _atmosphere(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nrlmsise00_density_geodetic, m)?)?;
    m.add_function(wrap_pyfunction!(nrlmsise00_density, m)?)?;
    m.add_function(wrap_pyfunction!(atmospheric_density, m)?)?;
    m.add_function(wrap_pyfunction!(co_rotating_atmosphere_velocity, m)?)?;
    m.add_class::<AtmosphericModel>()?;
    m.add_class::<NRLMSISE00>()?;
    m.add("NRMLSISE", m.getattr("NRLMSISE00")?)?;
    Ok(())
}
