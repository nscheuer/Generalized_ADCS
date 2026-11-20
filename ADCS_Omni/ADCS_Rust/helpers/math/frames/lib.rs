use nalgebra::{Matrix3, SVector};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;
type V = [f64; 3];
type M = Matrix3<f64>;

include!("common.rs");
include!("frames.rs");
include!("conversions.rs");

#[pymodule]
fn _frames(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! add{($($f:ident),*)=>{$(m.add_function(wrap_pyfunction!($f,m)?)?;)*}}
    add!(
        inertial_to_body_matrix,
        body_to_inertial_matrix,
        inertial_to_body,
        body_to_inertial,
        transform_vector,
        transform_covariance,
        gcrf_to_lvlh_matrix,
        gcrf_to_ecef_matrix,
        ecef_to_gcrf_matrix,
        gcrf_to_teme_matrix,
        teme_to_gcrf_matrix,
        teme_to_ecef_matrix,
        ecef_to_teme_matrix,
        frame_rotation,
        convert_vector,
        ecef_to_geodetic,
        gcrf_position_to_geodetic
    );
    Ok(())
}
