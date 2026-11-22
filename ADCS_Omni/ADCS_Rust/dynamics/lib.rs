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
