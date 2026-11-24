use crate::srp_model::{
    projected_area_srp_force as native_force,
    projected_area_srp_torque as srp_native_torque,
};

fn vectors3(rows: Vec<Vec<f64>>, name: &str) -> PyResult<Vec<Vector3>> {
    rows.into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|_| PyValueError::new_err(format!("{name} must have 3 columns")))
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature=(position_i,sun_position_i,q_b_from_i,normals_b,areas,optical_coefficients,illumination=1.0))]
pub fn projected_area_srp_force(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    q_b_from_i: &PyAny,
    normals_b: &PyAny,
    areas: &PyAny,
    optical_coefficients: &PyAny,
    illumination: f64,
) -> PyResult<PyObject> {
    let position = vec3(py, position_i, "position_i")?;
    let sun = vec3(py, sun_position_i, "sun_position_i")?;
    let quaternion = quat(py, q_b_from_i)?;
    let normals = vectors3(rows(py, normals_b, "normals_b", 3)?, "normals_b")?;
    let areas = vals(py, areas, "areas")?;
    let optical = vectors3(
        rows(py, optical_coefficients, "optical_coefficients", 3)?,
        "optical_coefficients",
    )?;
    let force = native_force(
        position,
        sun,
        quaternion,
        &normals,
        &areas,
        &optical,
        illumination,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out(py, force))
}

#[pyfunction]
#[pyo3(signature=(position_i,sun_position_i,q_b_from_i,normals_b,arms_b,areas,optical_coefficients,illumination=1.0))]
pub fn projected_area_srp_torque(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    q_b_from_i: &PyAny,
    normals_b: &PyAny,
    arms_b: &PyAny,
    areas: &PyAny,
    optical_coefficients: &PyAny,
    illumination: f64,
) -> PyResult<PyObject> {
    let position = vec3(py, position_i, "position_i")?;
    let sun = vec3(py, sun_position_i, "sun_position_i")?;
    let quaternion = quat(py, q_b_from_i)?;
    let normals = vectors3(rows(py, normals_b, "normals_b", 3)?, "normals_b")?;
    let arms = vectors3(rows(py, arms_b, "arms_b", 3)?, "arms_b")?;
    let areas = vals(py, areas, "areas")?;
    let optical = vectors3(
        rows(py, optical_coefficients, "optical_coefficients", 3)?,
        "optical_coefficients",
    )?;
    let torque = srp_native_torque(
        position,
        sun,
        quaternion,
        &normals,
        &arms,
        &areas,
        &optical,
        illumination,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out(py, torque))
}
