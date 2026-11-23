#![allow(non_local_definitions)]
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

include!("_geometry.rs");
include!("drag_model/projected_area.rs");
include!("srp_model/projected_area.rs");
include!("gravity_gradient_model/static.rs");
include!("residual_dipole_model/static.rs");
include!("third_body_model/lunisolar.rs");
include!("zonal_model/zonal.rs");

#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/drag_model/test_projected_area.rs"]
mod test_drag_model;
#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/gravity_gradient_model/test_static.rs"]
mod test_gravity_gradient_model;
#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/residual_dipole_model/test_static.rs"]
mod test_residual_dipole_model;
#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/srp_model/test_projected_area.rs"]
mod test_srp_model;
#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/third_body_model/test_lunisolar.rs"]
mod test_third_body_model;
#[cfg(test)]
#[path = "../../../tests/backends/rust/dynamics/zonal_model/test_zonal.rs"]
mod test_zonal_model;

#[pymodule]
fn _dynamics(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(projected_area_drag_force, m)?)?;
    m.add_function(wrap_pyfunction!(projected_area_drag_torque, m)?)?;
    m.add_function(wrap_pyfunction!(projected_area_srp_force, m)?)?;
    m.add_function(wrap_pyfunction!(projected_area_srp_torque, m)?)?;
    m.add_function(wrap_pyfunction!(static_residual_dipole_torque, m)?)?;
    m.add_function(wrap_pyfunction!(static_gravity_gradient_torque, m)?)?;
    m.add_function(wrap_pyfunction!(lunisolar_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(zonal_gravity_force, m)?)?;
    Ok(())
}
