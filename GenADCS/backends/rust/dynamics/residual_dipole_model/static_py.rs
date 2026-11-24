use crate::residual_dipole_model::static_residual_dipole_torque as residual_native_torque;

#[pyfunction]
pub fn static_residual_dipole_torque(
    py: Python<'_>,
    dipole_b: &PyAny,
    field_i: &PyAny,
    q_b_from_i: &PyAny,
) -> PyResult<PyObject> {
    let dipole = vec3(py, dipole_b, "dipole_b")?;
    let field = vec3(py, field_i, "field_i")?;
    let quaternion = quat(py, q_b_from_i)?;
    let torque = residual_native_torque(dipole, field, quaternion)
        .map_err(PyValueError::new_err)?;
    Ok(out(py, torque))
}
