#![allow(non_local_definitions)]
use pyo3::prelude::*;
pub mod satellite;
use satellite::Satellite as NativeSatellite;
mod satellite_py;
use satellite_py::PySatellite;
#[pymodule]
fn _satellite(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySatellite>()?;
    Ok(())
}
