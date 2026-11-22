#![allow(non_local_definitions)]
use numpy::ToPyArray;
use pyo3::prelude::*;

include!("j_gravity_model.rs");
include!("j2.rs");
include!("j3.rs");
include!("j4.rs");
include!("j5.rs");
include!("j6.rs");

#[pymodule]
fn _gravity(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add(
        "EARTH_ZONAL_COEFFICIENTS",
        EARTH_ZONAL_COEFFICIENTS.to_pyarray(py),
    )?;
    m.add("EARTH_MU_M3_S2", EARTH_MU_M3_S2)?;
    m.add("EARTH_REFERENCE_RADIUS_M", EARTH_REFERENCE_RADIUS_M)?;
    m.add_function(wrap_pyfunction!(gravity_acceleration_inertial_m_s2, m)?)?;
    m.add_class::<JGravityModel>()?;
    m.add_class::<J2>()?;
    m.add_class::<J3>()?;
    m.add_class::<J4>()?;
    m.add_class::<J5>()?;
    m.add_class::<J6>()?;
    Ok(())
}
