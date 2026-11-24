#[pyfunction]
fn lunisolar_acceleration(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    moon_position_i: &PyAny,
    sun_mu: f64,
    moon_mu: f64,
) -> PyResult<PyObject> {
    let position_i = vec3(py, position_i, "position_i")?;
    let sun_position_i = vec3(py, sun_position_i, "sun_position_i")?;
    let moon_position_i = vec3(py, moon_position_i, "moon_position_i")?;
    let result =
        lunisolar_acceleration_native(position_i, sun_position_i, moon_position_i, sun_mu, moon_mu)
            .map_err(PyValueError::new_err)?;
    Ok(out(py, result))
}
