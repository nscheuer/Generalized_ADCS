#[pyfunction]
fn zonal_gravity_force(
    py: Python<'_>,
    position_i: &PyAny,
    mass: f64,
    epoch_jd: f64,
    gravitational_parameter: f64,
    reference_radius: f64,
    zonal_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let position_i = vec3(py, position_i, "position_i")?;
    let zonal_coefficients = vals(py, zonal_coefficients, "zonal_coefficients")?;
    let force = zonal_gravity_force_native(
        position_i,
        mass,
        epoch_jd,
        gravitational_parameter,
        reference_radius,
        &zonal_coefficients,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out(py, force))
}
