#[pyfunction]
fn static_gravity_gradient_torque(
    py: Python<'_>,
    position_i: &PyAny,
    q_b_from_i: &PyAny,
    inertia_b: &PyAny,
    gravitational_parameter: f64,
) -> PyResult<PyObject> {
    let p = vec3(py, position_i, "position_i")?;
    let r = norm(p);
    if r == 0. {
        return Err(PyValueError::new_err(
            "gravity gradient is undefined at the body's centre",
        ));
    }
    let radial = mv(rotation(quat(py, q_b_from_i)?), scale(p, 1. / r));
    let inertia = mat3(py, inertia_b, "inertia_b")?;
    let ir = mv(inertia, radial);
    Ok(out(
        py,
        scale(cross(radial, ir), 3. * gravitational_parameter / r.powi(3)),
    ))
}
