#![allow(non_local_definitions)]
use pyo3::prelude::*;

include!("satellite_panel.rs");
include!("satellite_geometry.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/satellite_geometry/test_satellite_geometry.rs"]
mod test_satellite_geometry;

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/satellite_geometry/test_satellite_panel.rs"]
mod test_satellite_panel;

#[pymodule]
fn _geometry(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SatellitePanel>()?;
    m.add_class::<SatelliteGeometry>()?;
    Ok(())
}
