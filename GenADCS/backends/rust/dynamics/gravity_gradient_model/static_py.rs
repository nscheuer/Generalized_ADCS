use super::super::*;

#[pyfunction]
pub fn static_gravity_gradient_torque(
    py: Python<'_>,
    position_i: &PyAny,
    q_b_from_i: &PyAny,
    inertia_b: &PyAny,
    gravitational_parameter: f64,
) -> PyResult<PyObject> {
    let p = vec3(py, position_i, "position_i")?;
    let q = quat(py, q_b_from_i)?;
    let inertia = mat3(py, inertia_b, "inertia_b")?;
    let torque =
        super::static_model::static_gravity_gradient_torque(p, q, inertia, gravitational_parameter)
            .map_err(PyValueError::new_err)?;
    Ok(out(py, torque))
}
