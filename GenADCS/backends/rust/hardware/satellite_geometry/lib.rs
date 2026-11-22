#![allow(non_local_definitions)]
use pyo3::prelude::*;

include!("satellite_panel.rs");
include!("satellite_geometry.rs");

#[pymodule]
fn _geometry(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SatellitePanel>()?;
    m.add_class::<SatelliteGeometry>()?;
    Ok(())
}
