#![allow(non_local_definitions)]
use numpy::ToPyArray;
use pyo3::prelude::*;

include!("j_gravity_model.rs");
include!("j_gravity_model_py.rs");
include!("j2.rs");
include!("j3.rs");
include!("j4.rs");
include!("j5.rs");
include!("j6.rs");
include!("j2_py.rs");
include!("j3_py.rs");
include!("j4_py.rs");
include!("j5_py.rs");
include!("j6_py.rs");

#[cfg(test)]
#[path = "../../../../tests/backends/rust/environment/gravity/test_gravity.rs"]
mod tests;

#[pymodule]
fn _gravity(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add(
        "EARTH_ZONAL_COEFFICIENTS",
        EARTH_ZONAL_COEFFICIENTS.to_pyarray(py),
    )?;
    m.add("EARTH_MU_M3_S2", EARTH_MU_M3_S2)?;
    m.add("EARTH_REFERENCE_RADIUS_M", EARTH_REFERENCE_RADIUS_M)?;
    m.add_function(wrap_pyfunction!(gravity_acceleration_inertial_m_s2, m)?)?;
    m.add_class::<JGravityModelPy>()?;
    m.add_class::<J2Py>()?;
    m.add_class::<J3Py>()?;
    m.add_class::<J4Py>()?;
    m.add_class::<J5Py>()?;
    m.add_class::<J6Py>()?;
    Ok(())
}
