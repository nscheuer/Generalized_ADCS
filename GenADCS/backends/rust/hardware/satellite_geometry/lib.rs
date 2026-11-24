#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod satellite_geometry;
mod satellite_geometry_py;
pub mod satellite_panel;
mod satellite_panel_py;

pub use satellite_geometry::SatelliteGeometry;
pub use satellite_panel::SatellitePanel;

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/satellite_geometry/test_satellite_geometry.rs"]
mod test_satellite_geometry;

#[cfg(test)]
#[path = "../../../../tests/backends/rust/hardware/satellite_geometry/test_satellite_panel.rs"]
mod test_satellite_panel;

#[pymodule]
fn _geometry(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<satellite_panel_py::SatellitePanel>()?;
    m.add_class::<satellite_geometry_py::SatelliteGeometry>()?;
    Ok(())
}
