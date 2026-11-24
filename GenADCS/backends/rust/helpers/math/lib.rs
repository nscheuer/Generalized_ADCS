#![allow(non_local_definitions)]
use nalgebra::{Matrix3, SVector};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
pub type M = Matrix3<f64>;

include!("common.rs");
include!("common_py.rs");
include!("rotations/quaternion/quaternion.rs");
include!("rotations/quaternion/quaternion_py.rs");
include!("rotations/quaternion/algebra.rs");
include!("rotations/quaternion/algebra_py.rs");
include!("rotations/quaternion/conversions.rs");
include!("rotations/quaternion/conversions_py.rs");
include!("rotations/quaternion/interpolation.rs");
include!("rotations/quaternion/interpolation_py.rs");
include!("rotations/dcm/dcm.rs");
include!("rotations/dcm/dcm_py.rs");
include!("rotations/mrp/modified_rodrigues_parameters.rs");
include!("rotations/mrp/modified_rodrigues_parameters_py.rs");
include!("rotations/rodrigues/rodrigues_parameters.rs");
include!("rotations/rodrigues/rodrigues_parameters_py.rs");
include!("vector.rs");
include!("vector_py.rs");
include!("rotations/dcm/kernels.rs");
include!("rotations/dcm/kernels_py.rs");
include!("rotations/mrp/kernels.rs");
include!("rotations/mrp/kernels_py.rs");
include!("rotations/rodrigues/kernels.rs");
include!("rotations/rodrigues/kernels_py.rs");
include!("derivatives.rs");
include!("derivatives_py.rs");
include!("dynamics.rs");
include!("dynamics_py.rs");
include!("time/conversions.rs");
include!("time/conversions_py.rs");

#[pymodule]
fn _helpers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! add {($($f:ident),* $(,)?)=>{$(m.add_function(wrap_pyfunction!($f,m)?)?;)*};}
    add!(
        cross,
        norm,
        normalize,
        unit,
        skew,
        _normalize_quaternion,
        _quaternion_norm,
        _jpl_to_hamilton,
        _hamilton_to_jpl,
        _conjugate,
        _inverse,
        _multiply,
        _relative_quaternion,
        _rotate_vector,
        _quaternion_to_dcm,
        _dcm_to_quaternion,
        _axis_angle_to_quaternion,
        rotation_vector_to_quaternion,
        quaternion_to_rotation_vector,
        slerp,
        axis_angle_to_dcm,
        rotate_vector,
        inverse_dcm,
        compose_dcm,
        relative_dcm,
        dcm_orthogonality_error,
        normalize_dcm,
        mrp_to_quaternion,
        quaternion_to_mrp,
        mrp_shadow_set,
        rodrigues_to_quaternion,
        quaternion_to_rodrigues,
        quaternion_derivative,
        quaternion_rate_matrix,
        angular_velocity_from_quaternion_derivative,
        quaternion_normalization_jacobian,
        rotated_vector_quaternion_jacobian,
        quaternion_to_euler321,
        rigid_body_angular_acceleration,
        wheel_coupled_angular_acceleration,
        wheel_momentum_derivative,
        calendar_to_jd,
        jd_to_calendar,
        utc_jd_to_tai_jd,
        tai_jd_to_utc_jd,
        tai_jd_to_tt_jd,
        tt_jd_to_tai_jd,
        utc_jd_to_tt_jd,
        tt_jd_to_utc_jd,
        utc_jd_to_ut1_jd,
        ut1_jd_to_utc_jd,
        tt_jd_to_tdb_jd,
        tdb_jd_to_tt_jd,
        utc_jd_to_tdb_jd,
        tdb_jd_to_utc_jd,
        tai_jd_to_ut1_jd,
        ut1_jd_to_tai_jd,
        tt_jd_to_ut1_jd,
        ut1_jd_to_tt_jd,
        tai_jd_to_tdb_jd,
        tdb_jd_to_tai_jd,
        ut1_jd_to_tdb_jd,
        tdb_jd_to_ut1_jd,
        jd_to_utc_calendar_fields,
        day_of_year
    );
    m.add_class::<Quaternion>()?;
    m.add_class::<DCM>()?;
    m.add_class::<MRP>()?;
    m.add_class::<RotationParameter>()?;
    m.add(
        "Modified_Rodrigues_Parameters",
        m.getattr("Modified_Rodriguez_Parameters")?,
    )?;
    m.add(
        "ModifiedRodriguesParameters",
        m.getattr("Modified_Rodriguez_Parameters")?,
    )?;
    m.add_class::<Rodrigues>()?;
    m.add("Rodrigues_Parameters", m.getattr("Rodriguez_Parameters")?)?;
    m.add("RodriguesParameters", m.getattr("Rodriguez_Parameters")?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/rotations/dcm/test_dcm.rs"]
mod test_rotation_dcm;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/rotations/mrp/test_mrp.rs"]
mod test_rotation_mrp;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/rotations/quaternion/test_quaternion.rs"]
mod test_rotation_quaternion;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/rotations/rodrigues/test_rodrigues.rs"]
mod test_rotation_rodrigues;

#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/test_derivatives.rs"]
mod test_derivatives;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/test_dynamics.rs"]
mod test_dynamics;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/time/test_conversions.rs"]
mod test_time_conversions;
#[cfg(test)]
#[path = "../../../../tests/backends/rust/helpers/math/test_vector.rs"]
mod test_vector;
