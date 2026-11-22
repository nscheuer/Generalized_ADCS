#[pyfunction]
fn static_residual_dipole_torque(
    py: Python<'_>,
    dipole_b: &PyAny,
    field_i: &PyAny,
    q_b_from_i: &PyAny,
) -> PyResult<PyObject> {
    let dipole = vec3(py, dipole_b, "dipole_b")?;
    let field = mv(
        rotation(quat(py, q_b_from_i)?),
        vec3(py, field_i, "field_i")?,
    );
    Ok(out(py, cross(dipole, field)))
}
