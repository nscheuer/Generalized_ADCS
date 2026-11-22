#[pyfunction]
fn lunisolar_acceleration(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    moon_position_i: &PyAny,
    sun_mu: f64,
    moon_mu: f64,
) -> PyResult<PyObject> {
    let p = vec3(py, position_i, "position_i")?;
    let mut result = [0.; 3];
    for (body, mu) in [
        (vec3(py, sun_position_i, "sun_position_i")?, sun_mu),
        (vec3(py, moon_position_i, "moon_position_i")?, moon_mu),
    ] {
        let rel = [body[0] - p[0], body[1] - p[1], body[2] - p[2]];
        let d = norm(body);
        let r = norm(rel);
        if d == 0. || r == 0. {
            return Err(PyValueError::new_err("third-body distance must be nonzero"));
        }
        result = add(
            result,
            scale(
                [
                    rel[0] / r.powi(3) - body[0] / d.powi(3),
                    rel[1] / r.powi(3) - body[1] / d.powi(3),
                    rel[2] / r.powi(3) - body[2] / d.powi(3),
                ],
                mu,
            ),
        );
    }
    Ok(out(py, result))
}
